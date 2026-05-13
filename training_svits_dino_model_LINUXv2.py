from omegaconf import OmegaConf
from datasets.dataloaders import get_dino_dataloader
from models.dino_enginev2 import DINOTrainerModel
import os
from utils.reporters import ReporterBase
import yaml
import torch
import pandas as pd
torch.cuda.empty_cache()

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def main():

    config_omg = OmegaConf.load('runs_dino/run4_48px_months12_DinoTSViT_dim256ViTcls/config.yml')

    #config_omg.GENERAL.output_path = 'runs_dino'

    #config_omg.GENERAL.output_path = os.path.join(config_omg.GENERAL.output_path, 'run4_{}px_months{}_{}_dim{}{}{}'.format(
    #    config_omg.MODEL.img_res,config_omg.DATASETS.n_months, config_omg.MODEL.model_name, config_omg.MODEL.dim, config_omg.MODEL.spatial_block,
    #    config_omg.MODEL.pool
    #        ))

    if os.path.exists(os.path.join(config_omg.GENERAL.output_path, 'reporter.json')):
        reporter = ReporterBase()
        reporter.set_reporter(['epoch', 'iter_in_epoch', 'global_iter', 'global_loss', 'local_loss'])
        reporter.load_reporter(os.path.join(config_omg.GENERAL.output_path, 'reporter.json'))
        reporter.file_name = 'reporter.json'

        
    else:
        reporter = None

    resume_from = os.path.join(config_omg.GENERAL.output_path, "TSViTSdino_checkpoint_iter17135_ep20.pth")
    #torch.load('runs_dino/run4_48px_months12_DinoTSViT_dim256ViTcls/TSViTSdino_checkpoint_iter17135_ep20.pth', map_location=self.device, weights_only = False)
    print(config_omg.GENERAL.output_path)

    dataloader = get_dino_dataloader(config_omg.DATASETS.paths.training_input, batch_size=config_omg.DATASETS.bach_size, n_months=config_omg.DATASETS.n_months,
                        n_bands=config_omg.DATASETS.n_bands, img_size = config_omg.MODEL.img_res)
    
    if not os.path.exists(config_omg.GENERAL.output_path): os.makedirs(config_omg.GENERAL.output_path)
    
    #config_omg.MODEL.pretrained = 'runs_dino/run2_48px_months12_TSViT_single_token_dim256ViTmean/TSViTSdino_20_model_params'
    dinomodel = DINOTrainerModel(config_omg, dataloader, 
    model_weight_path = config_omg.GENERAL.output_path, reporter = None)

    #optimizer_to(dinomodel.optimizer,dinomodel.device)
    dinomodel.to(dinomodel.device)
    with open(os.path.join(config_omg.GENERAL.output_path,'config.yml'), 'w') as outfile:
        yaml.dump(OmegaConf.to_container(config_omg, resolve=True), outfile, default_flow_style=False)
    #resume_from = None
    dinomodel.fit(max_epochs=config_omg.TRAIN.max_epochs, 
                resume_from_checkpoint = resume_from,
                start_saving_from = config_omg.TRAIN.start_saving_from)

if __name__ == '__main__':
    main()

    
        