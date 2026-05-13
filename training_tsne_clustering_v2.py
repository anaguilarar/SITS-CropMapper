import rioxarray as rio
import os
from omegaconf import OmegaConf
import xarray


from detection.dataset import InputImgDataset
from detection.detectors import STViTS_detector


from skimage.transform import resize

from detection.utils import predict_tile, DAYS_IN_MONTH
import numpy as np
import torch
from scipy import stats
from tqdm import tqdm
import random

from sklearn.cluster import KMeans

from openTSNE import TSNE
import pickle

def mlt_prediction_tile(tile_id, crop_detector, data_loader, ending_year, ending_month, n_months, use_summary_layer = True):
    month = ending_month
    year = ending_year
    predictionpermonth = []
    for _ in range(n_months):

        if month == 0:
            month = 12
            year -= 1
        if month == -1 and month %2 != 0:
            month = 11
        day = DAYS_IN_MONTH[month]
        monthstr = str(month) if month>=10 else f'0{month}'
        date = f'{year}-{monthstr}-{day}'
        
        xr_prediction = predict_tile(tile_id, crop_detector, data_loader, date, use_summary_layer)
        predictionpermonth.append(xr_prediction.values)
        
        month -= 1
        
    return predictionpermonth

def mlt_tokens(tile_id, crop_detector, data_loader, ending_year, ending_month, n_months):
    month = ending_month
    year = ending_year
    tokenspermonth = []
    for _ in tqdm(range(n_months)):

        if month == 0:
            month = 12
            year -= 1
        if month == -1 and month %2 != 0:
            month = 11
        day = DAYS_IN_MONTH[month]
        monthstr = str(month) if month>=10 else f'0{month}'
        date = f'{year}-{monthstr}-{day}'
        img, _ = data_loader.__getitem__(tile_id, starting_date=None, ending_date = date, scale = True, reference_date = np.array(date).astype('datetime64[D]'))
        
        with torch.no_grad():
            predicted_tokens = crop_detector.intermediate_spatial_features(img.unsqueeze(0))
            predicted_tokens = crop_detector.model.backbone.norm_final(predicted_tokens)
        numpatches = crop_detector.model.backbone.image_size//crop_detector.model.backbone.patch_size
        
        tokenspermonth.append(predicted_tokens.reshape(1* numpatches * numpatches, 
                            crop_detector.model.backbone.dim).detach().numpy())
        
        month -= 1
        
    return tokenspermonth


def main():
    config = OmegaConf.load('runs_dino_seg/run2_48px_months12_DinoTSViTViTsummconv2_lasttwounf/config.yml')
    config.DATASETS.paths.training_input = 'olancho48/all_filtered'
    
    ## loading model
    img_crop_detector = STViTS_detector(config, init_scheduler = False)

    weight_path = 'runs_dino_seg/run2_48px_months12_DinoTSViTViTsummconv2_lasttwounf/TSViTS_checkpoint_iter34314_ep83.pth'
    img_crop_detector.load_weights_for_detection(weight_path)

    imageloader = InputImgDataset(config.DATASETS.paths.training_input,n_months= config.DATASETS.n_months, n_bands=config.DATASETS.n_bands, 
                            img_size=config.MODEL.img_res, summarize_img = config.TRAIN.get('summary_layer', False))

    imageloader6months = InputImgDataset(config.DATASETS.paths.training_input,n_months= 6, n_bands=config.DATASETS.n_bands, 
                            img_size=config.MODEL.img_res, summarize_img = config.TRAIN.get('summary_layer', False))

    processed_tiles = []
    tokens_sample = []
    for _ in tqdm(range(imageloader.__len__()//3)):
        try:
            tile_id = random.randint(0,imageloader.__len__())
            if tile_id not in processed_tiles:
                processed_tiles.append(tile_id)
                tokens_sample.append(
                mlt_tokens(tile_id, img_crop_detector, imageloader, 2024, ending_month=12, n_months=48))
        except:
            continue
        
    all_tokens = []
    for tk in tokens_sample:
        for tkt in tk:
            all_tokens.append(tkt)
            
    tokensda = np.array(all_tokens).reshape(len(all_tokens),24,24,256).swapaxes(0,1).swapaxes(1,2).reshape(24,24,len(all_tokens)*256)
    tokensda = resize(tokensda, (48,48), order=3, preserve_range=True, anti_aliasing=True).astype(float)

    tokensda = tokensda.reshape(48,48, len(all_tokens), 256)
    total_features = tokensda.reshape( 48 * 48 * tokensda.shape[2], tokensda.shape[3])#.reshape(4 * patch_h * patch_w, feat_dim) #4(*H*w, 1024)

    tsne = TSNE(
        perplexity=30,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=True,
    )
    
    embedding_train = tsne.fit(total_features)

    with open('tokens_tsne.pkl', 'wb') as f:
        pickle.dump(embedding_train, f)
        
        
if __name__ == "__main__":
    main()