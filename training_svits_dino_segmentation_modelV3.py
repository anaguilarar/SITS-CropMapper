from datasets.dataloaders import SegSatImgDataset, get_segmentation_dataloader, SegTileSatImgDataset
from datasets.transforms.image_transforms import ImageAugmentation, MTSegmenTransformer
from datasets.agro_satdata import GrowingSason_SatData, MltTileData
#import matplotlib.pyplot as plt
#from utils.plots import plot_multichanels
from datasets.transforms import HLSSCALERPARMS_7C
from omegaconf import OmegaConf
import numpy as np
import random
import os
from models.loss_functions import MaskedFocalLoss,FocalLoss
from models.dino_enginev2 import DINOTrainerSegmentationModel
from models.engine import optimizer_to
import yaml
from utils.reporters import ReporterBase
from datasets.utils import MaskingGenerator


def main():
    
    config = OmegaConf.load('runs_dino_seg/run2_48px_months12_DinoTSViTViTsummconv2_lasttwounf/config.yml')

    image_transformer = ImageAugmentation(min_max_parameters=config.DATASETS.transform_parameters)

    mlt_transformer = MTSegmenTransformer(image_transformer, 
            available_transforms=list(config.DATASETS.transform_parameters.keys()))
    
    mask_generator = MaskingGenerator(
            input_size=(config.MODEL.img_res, config.MODEL.img_res),
            max_num_patches=0.5 * config.MODEL.img_res * config.MODEL.img_res,
        )

    ## datasets
    tr_sat_dataset = get_segmentation_dataloader(config.DATASETS.paths.training_input, 
                                            config.DATASETS.paths.training_target, 
                                            aug_transform=mlt_transformer, 
                                            batch_size = config.DATASETS.bach_size,  
                                            n_months= config.DATASETS.n_months,
                                            n_bands=config.DATASETS.n_bands, 
                                            img_size=config.MODEL.img_res,
                                            mask_generator=mask_generator,
                                            mask_times=config.DATASETS.mask_times,
                                            summarize_img = config.TRAIN.summary_layer,
                                            tile_patches= None)

    val_sat_dataset = get_segmentation_dataloader(config.DATASETS.paths.validation_input, 
                                            config.DATASETS.paths.validation_target, 
                                            n_months= config.DATASETS.n_months,
                                            aug_transform=None, 
                                            batch_size = config.DATASETS.bach_size, 
                                            n_bands=config.DATASETS.n_bands,
                                            summarize_img = config.TRAIN.summary_layer,
                                            img_size=config.MODEL.img_res)
        


    reporter = ReporterBase()
    #if os.path.exists('runs_dino_seg/run_48px_months12_TSViT_single_tokenViTlogits/reporter.json')
    
    #config.TRAIN.output_path = os.path.join('runs_dino_seg', 
    #    'run2_{}px_months{}_{}{}{}2_lasttwounf'.format(
    #    config.MODEL.img_res,config.DATASETS.n_months, 
    #    config.MODEL.model_name, 
    #    config.MODEL.spatial_block, 
    #    config.MODEL.head_type
    #        ))
    reporter.file_name = 'reporter.json'
    if os.path.exists(os.path.join(config.TRAIN.output_path,reporter.file_name)):
        reporter.load_reporter(os.path.join(config.TRAIN.output_path,reporter.file_name))
        print('REPORTER LOADED')
       
    else:
        reporter.set_reporter(config.TRAIN.reporter_keys)
        checkpoint_path = None

    checkpoint_path = "/opt/suelos_honduras/crop_modeling/scripts/satellite_data/runs_dino_seg/run2_48px_months12_DinoTSViTViTsummconv2_lasttwounf/TSViTS_checkpoint_iter20859_ep50.pth"
    
        
    if not os.path.exists(config.TRAIN.output_path): os.makedirs(config.TRAIN.output_path)

    #config_omg.MODEL.num_channels = config_omg.DATASETS.n_bands+1
    ## lossess
    if config.LOSS.name == 'masked_focal_loss':
        loss_fn = MaskedFocalLoss(reduction='mean', gamma= config.LOSS.gamma, ignore_class=config.LOSS.exclude_class)
    elif config.LOSS.name == 'focal_loss':
        loss_fn = FocalLoss(reduction='mean', gamma= 1)
    elif config.LOSS.name == 'masked_cross_entropy':
        loss_fn = MaskedCrossEntropyLoss()

    ## model
    print(config.TRAIN.output_path)
    dino_seg_model = DINOTrainerSegmentationModel(config, 
    tr_sat_dataset, val_sat_dataset, 
        loss_fn = loss_fn,
        reporter= reporter, 
        reporter_losses= config.TRAIN.step_reporter_keys,
        model_weight_path = config.TRAIN.output_path,
        use_summary = config.TRAIN.summary_layer)
    
    optimizer_to(dino_seg_model.optimizer,dino_seg_model.device)
    dino_seg_model.to(dino_seg_model.device)

    dino_seg_model.model_name = config.MODEL.model_name + config.MODEL.spatial_block +'_'+config.MODEL.head_type
    
    with open(os.path.join(config.TRAIN.output_path,'config.yml'), 'w') as outfile:
        yaml.dump(OmegaConf.to_container(config, resolve=True), outfile, default_flow_style=False)
        

    dino_seg_model.fit(max_epochs=config.TRAIN.max_epochs,
    checkpoint_metric='eval_IOU',bestloss=0, 
    save_best = True, start_saving_from = 2,
    checkpoint = checkpoint_path)
    

if __name__ == '__main__':
    main()