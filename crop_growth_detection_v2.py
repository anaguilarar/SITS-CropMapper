
import rioxarray as rio
import os
from omegaconf import OmegaConf
import xarray

from detection.dataset import InputImgDataset
from detection.detectors import STViTS_detector

from scipy.signal import savgol_coeffs

from sklearn.cluster import KMeans

from detection.utils import predict_tile, DAYS_IN_MONTH
import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

from detection.smf_s_class import SMFS
from datasets.image_preprocessing import kernel_regression, chen_sg_filter
import random
from skimage.transform import resize
from openTSNE import TSNE
import pickle
import math

def pre_process_vi_ts_layer(vi_ts_layer,days, crop_mask = None, coeffs_trend1 = savgol_coeffs(11,7), coeffs_trend2 = savgol_coeffs(3,2 )):
    """
    vi_ts_layer: np.ndarray
    H, W, T
    """
    data2d = vi_ts_layer.reshape(vi_ts_layer.shape[0]*vi_ts_layer.shape[1],vi_ts_layer.shape[2])
    if crop_mask is not None:
        crop_mask2d = crop_mask.flatten()
    else:
        crop_mask2d = np.ones(data2d.shape[0], dtype=bool)
    #nonnanpos = np.unique(np.where(~np.isnan(data2d))[0])

    data2d_smoothed = np.zeros_like(data2d)
    for i in tqdm(range(data2d.shape[0])):
        yval = data2d[i]
        if np.all(np.isnan(yval)): continue
        if not crop_mask2d[i]: continue
        ts_interpolated = kernel_regression(yval, 0.03, days)
        ts_filtered = chen_sg_filter(ts_interpolated, coeffs_trend1 = coeffs_trend1, coeffs_trend2 = coeffs_trend2)
        data2d_smoothed[i] = ts_filtered

    data2d_smoothed[np.isnan(data2d_smoothed)] = 0        
    
    
    return data2d_smoothed.reshape(vi_ts_layer.shape[0],vi_ts_layer.shape[1],vi_ts_layer.shape[2])


def summarize_ts_per_cluster(vi_mlt_layer,cluster_labels, summarize_by = 'median'):
    if summarize_by == 'median':
        fun = np.nanmedian
    else:
        fun = np.mean
    vi_ts_cluster = {}
    for clv in np.unique(cluster_labels):
        datasub = np.zeros_like(vi_mlt_layer)
        datasub[:] = np.nan
        datasub[cluster_labels == clv] = vi_mlt_layer[cluster_labels == clv]
        datasub2d = datasub.reshape(datasub.shape[0]*datasub.shape[1],datasub.shape[2])
        
        vi_ts_cluster[clv] = fun(datasub2d[~np.any(np.isnan(datasub2d), axis = 1)], 0)
    
    return vi_ts_cluster

def find_phenology_per_cluster(pred_phen_days, reference_dates):
    
    pred_phen_date = {'Greenup': None,
                        'Maturity': None,
                        'Senescence': None,
                        'Dormancy': None}

    for i, day in enumerate(pred_phen_days):
        if day != 0 and not np.isnan(day) :
            pred_phen_date[list(pred_phen_date.keys())[i]] = reference_dates[0] + np.timedelta64(int(day), 'D')
    
    return pred_phen_date


def get_julian_day(date:np.datetime64):
    
    year = date.astype('datetime64[Y]').astype(int) + 1970
    str_date_ref_date = f'{year}-01-01'
    julian_day = date.astype('datetime64[D]')  - np.array(str_date_ref_date).astype('datetime64[D]') 
    return int(julian_day / np.timedelta64(1, 'D'))


def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    Points must be iterables of the same length (lists, tuples, etc.).
    """
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimension")
    
    squared_diff = sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))
    return math.sqrt(squared_diff)

def finding_phenology_using_smfs(ts_list:dict, ts_reference:dict, sim_theshold:float = 0.2, distance:str = 'euclidean'):
    vi_ts_reference = ts_reference['vi_ts']
    phe_days_list = ts_reference['phen_days']
    
    phen_days_dict = {}
    ts_series = {}
    
    for k,vi_ts_values in ts_list.items():
        DOYs = np.arange(1, 185, 14)[:-1]
        
        k = int(k)
        if np.any(np.isnan(vi_ts_values)): continue
        if distance == 'euclidean':
            sim_values = [euclidean_distance(vi_ts_values, vi_ts_reference[i]) for i in range(len(vi_ts_reference)) ]
        elif distance == 'dtw':
            sim_values = [dtw(vi_ts_values, vi_ts_reference[i]) for i in range(len(vi_ts_reference)) ]

        if sim_values[np.argmin(sim_values)]>sim_theshold: continue
                    
        ref_phe_days = np.array(phe_days_list[np.argmin(sim_values)])
        ref_vi_ts = np.array(vi_ts_reference[np.argmin(sim_values)])
        
        smfsphen_days = np.zeros((4, 1,1)).astype(float)
        smfsphen_days[:] = np.nan
        ts_series[k] =  vi_ts_values
        
        for phe_i in range(ref_phe_days.shape[0]): # Detect EMG, GUD and MAT
            #smfs_model = SMFS(evi_fts_ref, phe_ref[phe_i], DOYs)
            smfs_model = SMFS(ref_vi_ts, ref_phe_days[phe_i], DOYs)
            smfsphen_days[phe_i, 0,0] = smfs_model.doit(np.copy(vi_ts_values))
            
        phen_days_dict[k] = smfsphen_days
        
    return ts_series, phen_days_dict
    
def read_txt_files(path):
    file_content = []
    with open(path, "r") as fn:
        for line in fn.readlines():
            file_content.append(float(line))
    return file_content


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



def evi(npvalues):
    return 2.5 * (npvalues[3] - npvalues[2])/(npvalues[3] + 6*npvalues[2] - 7.5*npvalues[0] + 1)

def get_vi_image_ts(mlt_image:np.ndarray, vi:str = 'evi'):
    
    vinp_values = np.zeros_like(mlt_image[:,0].squeeze().swapaxes(0,1).swapaxes(1,2))
    for i in range(vinp_values.shape[-1]):
        if vi == 'evi':
            vinp_values[:,:,i] = evi(mlt_image[i])
        elif vi == 'ndvi':
            vinp_values[:,:,i] = mlt_image[i,-3]
    
    vinp_values[vinp_values == 0] = np.nan

    return vinp_values


def check_phenology(phen_array, diff_maxmaturity_greenup = 105, diff_dormancy_senescence = 105):
    def get_booleans():
        return np.isnan(greenup), np.isnan(maturity), np.isnan(senescence), np.isnan(dormancy)
    
    greenup, maturity, senescence, dormancy = phen_array.squeeze()
    b_greenup, b_maturity, b_senescence, b_dormancy = get_booleans()
    corrected_phen = np.zeros_like(phen_array)

    if b_greenup and not (not b_maturity and not b_senescence):
        maturity = np.nan
        senescence = np.nan
    b_greenup, b_maturity, b_senescence, b_dormancy = get_booleans()
    if not b_greenup and not b_maturity:

        if (maturity- greenup)>diff_maxmaturity_greenup:
            
            greenup, maturity = np.nan, np.nan
        if greenup >= maturity:
            greenup, maturity = np.nan, np.nan
    b_greenup, b_maturity, b_senescence, b_dormancy = get_booleans()
    if (b_greenup and b_maturity) and (not b_senescence and b_dormancy):
        senescence = np.nan
    b_greenup, b_maturity, b_senescence, b_dormancy = get_booleans()
    if not b_senescence and not b_dormancy:
        if (dormancy- senescence)>diff_dormancy_senescence:
            senescence, dormancy = np.nan, np.nan
    
    for i, val in enumerate([greenup, maturity, senescence, dormancy]): corrected_phen[i] = val

    return corrected_phen

def phen_map_toweekly(map_layer, ndays = 14):
    
    twoweekssequence = [[(i-1)*ndays,i*ndays] for i in range(1, (365//ndays)+1)]
    for val in np.unique(map_layer):
        if np.isnan(val): continue
        newval = [nw[0]+(nw[-1]-nw[0])//2 for nw in twoweekssequence if (val>=nw[0] and val<nw[1])]
        if newval:
            map_layer[map_layer == val] = newval[0]

    for nval in np.unique(map_layer):
      
        if (np.sum(map_layer == nval)/(map_layer.shape[0]*map_layer.shape[1]))* 100 < 1:
            
            map_layer[map_layer == nval] = np.nan
    
    return map_layer

def main():
    

    lc = {
        'crops' : [11, 12,13,20],
        'trees' : [1,2,3,4, 7, 8],
        'water' : [17, 18],
        'soil'  : [16,15],
        'urban' : [14],
        'vegetation' : [21,5,6],
        'others' : [9, 10, 19]
    }

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


        # crop mask
    tiles_prediction = []
    ref_phen_path = "phen_patterns_"
    ref_phen_files = os.listdir(ref_phen_path)

    nfiles = len(ref_phen_files)//2

    vi_ts_lst = []
    phe_days_list = []

    for i in range(nfiles):

        phe_days_list.append(
            read_txt_files(os.path.join(ref_phen_path, f"ref_phe_6months_{i}.txt")))

        vi_ts_lst.append(
            read_txt_files(os.path.join(ref_phen_path, f"ref_vi_6months_{i}.txt")))

    
    
    print(imageloader6months.__len__())
    for tile_id in tqdm(range(imageloader6months.__len__())):
    #for tile_id in tqdm(range(35,37)):


        try:
            embedding_trained = None
            
            ts_reference = {
                'vi_ts' : vi_ts_lst,
                'phen_days': phe_days_list
            }
            data_predictions = mlt_prediction_tile(tile_id, img_crop_detector, imageloader, 2024, ending_month=12, n_months=24, use_summary_layer=config.TRAIN.get('summary_layer', False))
            print(data_predictions[0].shape)

            crop_mask = np.zeros((len(data_predictions), data_predictions[0].shape[0], data_predictions[0].shape[1]))
            crop_values = lc['crops']

            for i in range(len(data_predictions)):
                crop_mask[i] = np.isin(data_predictions[i], crop_values)

            crop_mask = stats.mode(crop_mask, axis = 0)[0]
            
            # cluster
            crop_mask = crop_mask.squeeze()
            nmonths_cluster = 12
            cluster_year = 2023
            cluster_month = 11
            nclusters = 120
            all_tokens = mlt_tokens(tile_id, img_crop_detector, imageloader, cluster_year, ending_month=cluster_month, n_months=nmonths_cluster)

            print('tokens obtained **********')
            tokensda = np.array(all_tokens).reshape(len(all_tokens),24,24,256).swapaxes(0,1).swapaxes(1,2).reshape(24,24,len(all_tokens)*256)
            tokensda = resize(tokensda, (48,48), order=3, preserve_range=True, anti_aliasing=True).astype(float)
            tokensda = tokensda.reshape(48,48, len(all_tokens), 256)
            print(tokensda.shape)
            total_features = tokensda.reshape( tokensda.shape[0] * tokensda.shape[1], tokensda.shape[2], tokensda.shape[3])#.reshape(4 * patch_h * patch_w, feat_dim) #4(*H*w, 1024)

            if embedding_trained is None: 
                total_features_masked = total_features[crop_mask.reshape(48*48)==1]
                total_features_masked_2d = total_features_masked.reshape(total_features_masked.shape[0]*total_features_masked.shape[1], total_features_masked.shape[2])
                print(total_features_masked_2d.shape)
                tsne = TSNE(
                    perplexity=30,
                    metric="euclidean",
                    n_jobs=8,
                    random_state=42,
                    verbose=True,
                )
                embedding_trained = tsne.fit(total_features_masked_2d)
            
            print('tsne **********')

            #
            total_features[crop_mask.reshape(48*48)==0] = 0
            total_features2d = total_features.reshape(total_features.shape[0]*total_features.shape[1],total_features.shape[2])
            tsnefeatures = embedding_trained.transform(total_features2d)
            
            kmeans = KMeans(n_clusters=nclusters, random_state=42, n_init="auto")
            kmeans.fit(tsnefeatures)
            labels = kmeans.predict(tsnefeatures)
            tsne_labels = {}

            year = cluster_year
            month = cluster_month

            for m in range(tokensda.shape[2]):
                if month == 0:
                    month = 12
                    year -= 1
                if month == -1 and month %2 != 0:
                    month = 11
                    year -= 1
                day = DAYS_IN_MONTH[month]
                monthstr = str(month) if month>=10 else f'0{month}'
                date = f'{year}-{monthstr}-{day}'
                month -=1
                
                labels = kmeans.predict(tsnefeatures.reshape([tokensda.shape[0]*tokensda.shape[1],tokensda.shape[2],2])[:,m])
                labels = labels.reshape(48,48).astype(float)
                
                tsne_labels[date] = labels


            phen_mlt = []
            maxmonths = 8
            processeddates = []
            print('Crop groth patterns matching **********')
            for date  in list(tsne_labels.keys())[:maxmonths]:
                
                labels = tsne_labels[date]
                
                img, _ = imageloader6months.__getitem__(tile_id, starting_date=None, ending_date = date, scale = False, reference_date = np.array(date).astype('datetime64[D]'))
                real_dates = imageloader6months._new_dates
            #    print(real_dates)
                vi_mlt_layer = get_vi_image_ts(img.detach().numpy(), vi = 'evi')
                days_vals = img.detach().numpy()[:,-1,0,0]
                data_smoothed = pre_process_vi_ts_layer(vi_mlt_layer, days_vals, crop_mask)
                vi_ts_cluster_list = summarize_ts_per_cluster(data_smoothed, labels)

                ts_series, phen_values = finding_phenology_using_smfs(vi_ts_cluster_list, ts_reference, sim_theshold= 0.22)
                if phen_values:
                    phen_values = {k: check_phenology(v) for k, v in phen_values.items()}
                    
                    phen_dates_percluster = {k: find_phenology_per_cluster(v, real_dates) for k, v in phen_values.items()}

                    data_smoothed_2d = data_smoothed.reshape(data_smoothed.shape[0]*data_smoothed.shape[1], data_smoothed.shape[2])
                    phen_value = np.zeros((4,data_smoothed.shape[0]*data_smoothed.shape[1]), dtype = float)

                    for z in range(data_smoothed_2d.shape[0]):
                        euc_dist = [euclidean_distance(data_smoothed_2d[z], v) for k,v in ts_series.items()]
                        min_pos = np.argmin(euc_dist)
                        phen_states_names = list(phen_dates_percluster[list(phen_dates_percluster.keys())[0]].keys())
                        for idx, phen_id in enumerate(phen_states_names):
                            phen_days = phen_dates_percluster[list(phen_dates_percluster.keys())[min_pos]][phen_id]
                            if euc_dist[min_pos] < 0.30 and phen_days is not None:
                                phen_value[idx,z] = get_julian_day(phen_days)
                                
                    phen_value = phen_value.reshape((phen_value.shape[0], 48,48))
                    phen_mlt.append(phen_value)
                    processeddates.append(date)
            median_phen_maps = np.zeros_like(phen_mlt[0])
            for j in range(median_phen_maps.shape[0]):
                green_upmap = np.array([phen_mlt[i][j] for i in range(len(phen_mlt))])
                green_upmap[green_upmap==0] = np.nan
                mapmedian = np.nanmedian(green_upmap, axis = 0)
                median_phen_maps[j] = phen_map_toweekly(mapmedian)
                
            phen_names = ['Greenup', 'Maturity', 'Senescence', 'Dormancy']
            bands = ['blue','green','red', 'nir']
            sp_data = xarray.zeros_like(imageloader6months._xrdata[bands].isel(date = 0))    
            for z in range(len(phen_names)): sp_data[bands[z]].values= median_phen_maps[z]
            sp_data = sp_data.rename({bb:pp for bb,pp in zip(bands, phen_names)})
            
            tiles_prediction.append(sp_data.drop('date'))
            print(tile_id != 0 and tile_id%5 ==0)
            if tile_id%5 ==0:
                print(' exporting temporal layer ')
                tile_crop_growth = xarray.merge(tiles_prediction)
                tile_crop_growth.attrs = None
                tile_crop_growth.rio.to_raster('tmp2.tif')
        except Exception as exc:
            print(f"Tile {tile_id} failed: {exc}")
            continue
    tile_crop_growth = xarray.merge(tiles_prediction)
    tile_crop_growth.attrs = None
    tile_crop_growth.rio.to_raster('all.tif')
    
    
if __name__ == '__main__':
    
    main()