from datasets.transforms.image_transforms import ImageAugmentation, MTSegmenTransformer
from utils.reporters import ReporterBase
from datasets.dataloaders import get_segmentation_dataloader

from models.engine import DLTrainerModel

from models.TSViT import architectures
from models.TSViT.swinTSViT import SwinTSViT, TSViT_Single_Token
from models.loss_functions import FocalLoss, MaskedFocalLoss
import torch
import torch.nn as nn

from omegaconf import OmegaConf
import numpy as np
import yaml
import os

CONFIG = {
    'DATASETS':
        {
            'paths': {
                'training_input': 'hls_data96/training_input/',
                'training_target': 'hls_data96/training_target/',
                'validation_input': 'hls_data96/validation_input/',
                'validation_target': 'hls_data96/validation_target/'
            },
            'transform_parameters': {
                #'denoise': [0.0001, 0.03],
                'gaussian': [0.0001, 0.03],
                'flip': [-1,1],
                'shift':[5, 20],
                'rotation': [10,180,],
                'zoom': [0.1, 0.5],
            },
            'bach_size': 8,
            'sobel': True,
            'n_months': 7
        },
    'MODEL':
        {'model_name': 'SwinTSViTS',
         'window_size': 4,
         'img_res': 96, 'patch_size': 3, 'patch_size_time': 1, 'patch_time': 4, 'num_classes': 22,
        'max_seq_len': 16, 'dim': 128, 'temporal_depth': 5, 'spatial_depth': 2, 'depth': 6,
        'heads': 4, 'pool': 'cls', 'num_channels': 6, 'dim_head': 64, 'dropout': 0, 'emb_dropout': 0.,
        'scale_dim': 4, "time_window": 151},
    'LOSS':
        {'half_T_side_dense': 10,
         'lamb': 1,
         'name': 'masked_focal_loss',
         'gamma': 1,
         'exclude_class': 0},
    'TRAINING':
        {'epochs': 200,
        'output_path': 'runs',
        'start_saving_from': 5,
        'reporter_keys': ['epoch', 
                       'train_loss', 'train_Accuracy', 'train_Precision', 'train_Recall', 'train_F1', 'train_IOU',
                        'eval_loss', 'eval_Accuracy', 'eval_Precision', 'eval_Recall', 'eval_F1', 'eval_IOU'],
        'step_reporter_keys': ['epoch', 'iter', 'loss', 'Accuracy', 'Precision', 'Recall', 'F1', 'IOU']}
        
    
}

def main():
    
    config_omg = OmegaConf.create(CONFIG)
    config_omg = OmegaConf.load('runs_test/config_LINUX.yml')
    config_omg.MODEL.time_window = (config_omg.DATASETS.n_months * 30)+1

    image_transformer = ImageAugmentation(min_max_parameters=config_omg.DATASETS.transform_parameters)
    mlt_transformer = MTSegmenTransformer(image_transformer, available_transforms=list(config_omg.DATASETS.transform_parameters.keys()))

    reporter = ReporterBase()
    if os.path.exists(os.path.join(config_omg.TRAINING.output_path,'reporter.json')):
        reporter.load_reporter(os.path.join(config_omg.TRAINING.output_path,'reporter.json'))
    else:
        reporter.set_reporter(config_omg.TRAINING.reporter_keys)
        config_omg.TRAINING.output_path = os.path.join(config_omg.TRAINING.output_path, 'run2_{}px_months{}_{}{}'.format(
        config_omg.MODEL.img_res,config_omg.DATASETS.n_months, config_omg.MODEL.model_name, config_omg.MODEL.head
            ))
    if not os.path.exists(config_omg.TRAINING.output_path): os.makedirs(config_omg.TRAINING.output_path)


    reporter.file_name = 'reporter.json'
    
    ## dataloaders

    tr_sat_dataset = get_segmentation_dataloader(config_omg.DATASETS.paths.training_input, config_omg.DATASETS.paths.training_target, n_months= config_omg.DATASETS.n_months,
                                            aug_transform=mlt_transformer, batch_size = config_omg.DATASETS.bach_size,  n_bands=config_omg.DATASETS.n_bands, img_size=config_omg.MODEL.img_res)
    val_sat_dataset = get_segmentation_dataloader(config_omg.DATASETS.paths.validation_input, config_omg.DATASETS.paths.validation_target, n_months= config_omg.DATASETS.n_months,
                                            aug_transform=None, batch_size = config_omg.DATASETS.bach_size,  n_bands=config_omg.DATASETS.n_bands, img_size=config_omg.MODEL.img_res)

    #config_omg.MODEL.num_channels = config_omg.DATASETS.n_bands+1
    ## model
    if config_omg.LOSS.name == 'masked_focal_loss':
        loss_fn = MaskedFocalLoss(reduction='mean', gamma= config_omg.LOSS.gamma, ignore_class=config_omg.LOSS.exclude_class)
    elif config_omg.LOSS.name == 'focal_loss':
        loss_fn = FocalLoss(reduction='mean', gamma= 1)
    elif config_omg.LOSS.name == 'masked_cross_entropy':
        loss_fn = MaskedCrossEntropyLoss()

    # increase the number of channels
    #if config_omg.DATASETS.sobel:
    #    config_omg.MODEL.num_channels = config_omg.MODEL.num_channels+2

    # newoutputpath 
    if config_omg.MODEL.model_name == 'SwinTSViTS':
        model = architectures.SwinTSViT(config_omg.MODEL)
    elif config_omg.MODEL.model_name == 'TSViTS':
        model = architectures.TSViT(config_omg.MODEL)
    elif config_omg.MODEL.model_name == 'TSViT_single_token':
        model = TSViT_Single_Token(config_omg.MODEL)
    
    model.model_name = config_omg.MODEL.model_name

    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4,
            weight_decay=0,

        )

    ## config export
    print(config_omg.TRAINING.output_path)
    with open(os.path.join(config_omg.TRAINING.output_path,'config.yml'), 'w') as outfile:
        yaml.dump(OmegaConf.to_container(config_omg, resolve=True), outfile, default_flow_style=False)
        
    trainer = DLTrainerModel(model, optimizer=optimizer, train_data_loader=tr_sat_dataset, validation_data_loader=val_sat_dataset, 
                            reporter=reporter, loss_fcn=loss_fn, reporter_losses = ['epoch', 'iter', 'loss', 'Accuracy', 'Precision', 'Recall', 'F1', 'IOU'],
                            model_weight_path=config_omg.TRAINING.output_path)
    trainer._nclasses = config_omg.MODEL.num_classes
    
    if os.path.exists(os.path.join(config_omg.TRAINING.output_path,'{}_18_model_params'.format(config_omg.MODEL.model_name))):
        trainer.load_weights({
            'model_path': os.path.join(config_omg.TRAINING.output_path,'{}_18_model_params'.format(config_omg.MODEL.model_name)),
            'optimizer_path': os.path.join(config_omg.TRAINING.output_path,'{}_18_optimizer_params'.format(config_omg.MODEL.model_name)),
            'scaler_path': os.path.join(config_omg.TRAINING.output_path,'{}_18_scaler_params'.format(config_omg.MODEL.model_name)),
          })
        

    trainer.fit(max_epochs=config_omg.TRAINING.epochs,checkpoint_metric='eval_IOU',best_value=0, save_best = True, start_saving_from = 2, start_from = 1)
    

if __name__ == '__main__':
    main()