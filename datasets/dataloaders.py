import os
from functools import partial
import random

from .agro_satdata import GrowingSason_SatData, MltTileData
from .transforms.tensor_transforms import ToTensor, Normalize, DataAugmentationDINO, DataAugmentationDINOV2
from .segmentation import TargetSegmentation
from .utils import MaskingGenerator, collate_data_and_cast, randomly_mask_times
    
import numpy as np    
import torch
from torch.utils.data import Dataset, DataLoader

class DinoSatTileDataset(Dataset, MltTileData):
    
    def __len__(self):
        
        
        return len(self.all_combinations)-2
    
        
    def __init__(self, nc_path, dino_transform = None, sobel_filter = True, 
                 n_months = 5, n_bands = 5, img_size = 96, sample = False, tile_patches = None):
        
        global_crops_scale = [0.32, 1.0]
        local_crops_number = 8
        local_crops_scale = [0.05, 0.32]

        global_crops_size = 48
        local_crops_size = 24

        self._n_months = n_months
        #files = list(set(os.path.join(nc_path, i) for i in os.listdir(nc_path) if i.endswith('.nc')))
        #self._raw_files = files
        self._img_size = img_size
        self.sample = sample
        Dataset.__init__(self)
        MltTileData.__init__(self, path = nc_path, 
                            months_season=self._n_months,
                            pixel_size = self._img_size,tile_patches=tile_patches)
        
        self.n_bands = n_bands
        self._nc_path = nc_path
    
        if dino_transform is None:
            self.dino_transform = DataAugmentationDINOV2(
            global_crops_scale,
            local_crops_scale,
            local_crops_number,
            std_range = (0.001,0.005),
            global_crops_size=global_crops_size,
            local_crops_size=local_crops_size,
            n_bands= n_bands
            )   
        else:
            self.dino_transform =dino_transform
        self.sobel_filter = sobel_filter
        
    def __getitem__(self, idx):
        
        satdata = self.get_random_date_tile_as_image(idx)
        augmenteddata = self.dino_transform(satdata)
        
        return augmenteddata

class DinoSatImgDataset(Dataset, GrowingSason_SatData):
    
    def __len__(self):
        
        
        return len(self._list_files)
    
        
    def __init__(self, nc_path, dino_transform = None, sobel_filter = True, n_months = 5, n_bands = 5, img_size = 96, sample = False):
        
        global_crops_scale = [0.32, 1.0]
        local_crops_number = 8
        local_crops_scale = [0.05, 0.32]

        global_crops_size = 48
        local_crops_size = 24

        self._n_months = n_months
        files = list(set(os.path.join(nc_path, i) for i in os.listdir(nc_path) if i.endswith('.nc')))
        self._raw_files = files
        self._img_size = img_size
        self.sample = sample
        Dataset.__init__(self)
        GrowingSason_SatData.__init__(self, list_files= self._raw_files, months_season=self._n_months, pixel_size = self._img_size )
        
        self.n_bands = n_bands
        self._nc_path = nc_path
    
        if dino_transform is None:
            self.dino_transform = DataAugmentationDINO(
            global_crops_scale,
            local_crops_scale,
            local_crops_number,
            global_crops_size=global_crops_size,
            local_crops_size=local_crops_size,
            n_bands= n_bands
            )   
        else:
            self.dino_transform =dino_transform
        self.sobel_filter = sobel_filter
        
    def __getitem__(self, idx):
        
        
        satdata = self.get_satellite_as_image(idx)
        augmenteddata = self.dino_transform(satdata)
        
        return augmenteddata
    
    

class SegSatImgDataset(Dataset):
    
    def __len__(self):
        
        
        return len(self.input_data._list_files)
    
    def get_files_subset(self):
        files = self._raw_files
        random.shuffle(files)
        return files[:int(len(files)*.15)]
        
    def __init__(self, nc_path, target_path, aug_transform = None, sobel_filter = True, n_months = 5, n_bands = 5, img_size = 96, sample = False):
        self._n_months = n_months
        files = list(set(os.path.join(nc_path, i) for i in os.listdir(nc_path) if i.endswith('.nc')))
        self._raw_files = files
        self._img_size = img_size
        self.sample = sample
        Dataset.__init__(self)
        if self.sample:
            self.input_data = GrowingSason_SatData(list_files= self.get_files_subset(), months_season=self._n_months, pixel_size = self._img_size )
        else:
            self.input_data = GrowingSason_SatData(list_files= self._raw_files, months_season=self._n_months, pixel_size = self._img_size )
            
        self.target_data = TargetSegmentation(pixel_size = self._img_size)
        self.n_bands = n_bands
        self._nc_path = nc_path
        self._target_path = target_path

        self.aug_transformer = aug_transform
        self.sobel_filter = sobel_filter
        
    def __getitem__(self, idx):
        
        
        satdata = self.input_data.get_satellite_as_image(idx)
        dates = satdata[:,-1]
        # target data
        filename = os.path.basename(self.input_data._tmp_file_path).split('.')[0]
        targetmask = self.target_data.get_target_image(filename=os.path.join(self._target_path,f'{filename}.tif'), invert_yaxis= self.input_data._invertyaxis)

        satdata = satdata[:,:-1]
        if self.aug_transformer is not None:
            satdata_g, targetmask_g = self.aug_transformer.transform_inputs(satdata.copy(), targetmask.copy())
        else:
            satdata_g, targetmask_g = satdata.copy(), targetmask.copy()
        
        satdata_g = ToTensor()(satdata_g)
    
        dates = torch.from_numpy(dates).unsqueeze(dim=1).to(torch.float32)
        
        satdata_g = torch.concat([satdata_g, dates], dim = 1)
        satdata_g = Normalize('hls', self.n_bands)(satdata_g)
        targetmask_g = torch.from_numpy(targetmask_g).unsqueeze(dim=0).to(torch.float32)
        if self.sample:
            self.input_data = GrowingSason_SatData(list_files= self.get_files_subset(), months_season=self._n_months, pixel_size = self._img_size)
        else:
            self.input_data = GrowingSason_SatData(list_files= self._raw_files, months_season=self._n_months, pixel_size = self._img_size)
            
        return satdata_g, targetmask_g
    

class IICSatImgDataset(Dataset, GrowingSason_SatData):
    
    def __len__(self):
        return len(self._list_files)
    
    def __init__(self, nc_path, aug_transform = None, affine_transform = None, n_months = 5, n_bands = 5, img_size = 96, sample = False):
        files = list(set(os.path.join(nc_path, i) for i in os.listdir(nc_path) if i.endswith('.nc')))
        self._n_months = n_months
        self._img_size = img_size
        self.n_bands = n_bands
        
        GrowingSason_SatData.__init__(self, list_files=files, months_season=self._n_months, pixel_size = self._img_size )
        self.affine_transform = affine_transform
        self.aug_transformer = aug_transform
        
    def __getitem__(self, idx):
        
        
        satdata = self.get_satellite_as_image(idx)
        dates = satdata[:,-1]
        
        satdata = satdata[:,:-1]
        satdata_g = self.aug_transformer(satdata.copy())
        
        #if  self.transform is not None:
        satdata = ToTensor()(satdata)
        satdata_g = ToTensor()(satdata_g)
        
        if self.affine_transform:
            satdata_g, img2_to_img1affine = self.affine_transform(satdata_g)
        else:
            img2_to_img1affine = None        
        
        dates = torch.from_numpy(dates).unsqueeze(dim=1).to(torch.float32)
        
        satdata = torch.concat([satdata, dates], dim = 1)
        satdata_g = torch.concat([satdata_g, dates], dim = 1)
        satdata = Normalize('hls', self.n_bands)(satdata)
        satdata_g = Normalize('hls', self.n_bands)(satdata_g)
        
        return satdata, satdata_g, img2_to_img1affine


def get_iic_dataloader(input_path,  batch_size=8, aug_transform = None, affine_transform = None, num_workers=0, shuffle=True,  n_months = 5, n_bands = 5, img_size = 96, sample = False):
    dataset = IICSatImgDataset(input_path, aug_transform=aug_transform, affine_transform= affine_transform,  n_months = n_months, n_bands = n_bands, img_size = img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

class SegTileSatImgDataset(Dataset):
    
    def __len__(self):
        
        return len(self.input_data.all_combinations)-2
    
    def get_files_subset(self):
        files = self._raw_files
        random.shuffle(files)
        return files[:int(len(files)*.15)]
        
    def __init__(self, input_path, target_path, aug_transform = None, n_months = 12, 
                n_bands = 7, img_size = 48, tile_patches = None, 
                mask_times = False, mask_generator = None,
                summarize_img = False):
        
        Dataset.__init__(self)
        self._mask_times = mask_times
        self.mask_generator = mask_generator
        self.summarize_img = summarize_img
        self.input_data = MltTileData(path = input_path,
                            months_season= n_months,
                            pixel_size = img_size, tile_patches=tile_patches)
                
        self.target_data = TargetSegmentation(pixel_size = img_size)
        self.n_bands = n_bands
        
        self._target_path = target_path

        self.aug_transformer = aug_transform
    
    def calculate_annual_summary(self, pixel_timeseries_np):
        # full_year_pixel_timeseries_np: (T_full_year, C_spectral) for one pixel

        valid_steps_mask = np.zeros_like(pixel_timeseries_np) # Mask for valid time steps
        valid_steps_mask[:] = pixel_timeseries_np
        valid_steps_mask[valid_steps_mask == 0] = np.nan

        median_summary = np.nanmedian(valid_steps_mask, axis = 0)
        median_summary[np.isnan(median_summary)] = 0# You could add other stats: np.median, np.std, np.percentile
            # summary_features = np.concatenate([mean_summary, median_summary, ...])
        return median_summary

        
    def __getitem__(self, idx):
        
        satdata = self.input_data.get_random_date_tile_as_image(idx)

        dates = satdata[:,-1]
        # target data
        filename = os.path.basename(self.input_data._tmp_file_path).split('.')[0]
        targetmask = self.target_data.get_target_image(filename=os.path.join(
            self._target_path,f'{filename}.tif'), invert_yaxis= self.input_data._invertyaxis)

        if self.aug_transformer is not None:
            satdata_g, targetmask_g = self.aug_transformer.transform_inputs(satdata[:,:-1].copy(), targetmask.copy())
        else:
            satdata_g, targetmask_g = satdata[:,:-1].copy(), targetmask.copy()
        
        t,c,h,w = satdata_g.shape
        n_tokens = h * w
        napos = np.where(np.all((satdata_g[:,0].reshape(t, h*w)==0), axis = 1))[0]
        #notnapos = [i for i in range(t) if i not in napos]
        if(len(napos)/t)<.4 and self._mask_times:
            satdata_g = randomly_mask_times(satdata_g, max_percentage=40)
        
        if self.mask_generator is not None:            
            masks_list = []
            for _ in range(t):
                masks_list.append(self.mask_generator(int(n_tokens * random.uniform(0, .15))))
            random.shuffle(masks_list)
            for i in range(len(masks_list)):
                satdata_g[i,:,masks_list[i]] = 0

        satdata_t = ToTensor()(satdata_g)
        dates = torch.from_numpy(dates).unsqueeze(dim=1).to(torch.float32)
        
        satdata_t = torch.concat([satdata_t, dates], dim = 1)
        satdata_t = Normalize('hls', self.n_bands)(satdata_t)
        targetmask_g = torch.from_numpy(targetmask_g).unsqueeze(dim=0).to(torch.float32)
        if self.summarize_img:
            img_summarized = self.calculate_annual_summary(satdata_g)
            dummytime = np.zeros((1,img_summarized.shape[-2],img_summarized.shape[-1])).astype(img_summarized.dtype)
            img_summarized = np.concatenate([img_summarized, dummytime], axis = 0) # C, H, W
            img_summarized_t = ToTensor()(img_summarized)
            img_summarized_t = Normalize('hls', self.n_bands)(img_summarized_t.unsqueeze(dim=0)).squeeze()
            return satdata_t, img_summarized_t[:-1], targetmask_g
        
        else:
            return satdata_t, targetmask_g    
    
def get_segmentation_dataloader(input_path,  target_path, batch_size=8, aug_transform = None, 
                                n_months = 5, n_bands = 5, img_size = 96, tile_patches = None, 
                                mask_generator=None, mask_times=True,
                                summarize_img = True):
    
    
    #mask_generator = MaskingGenerator(
    #        input_size=(img_size // patch_size, img_size // patch_size),
    #        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    #    )
    
    #ntokens = (img_size//patch_size)**2
    
    input_data = SegTileSatImgDataset(input_path=input_path, target_path = target_path, aug_transform = aug_transform, n_months=n_months, img_size = img_size, 
                            n_bands = n_bands, tile_patches = tile_patches, mask_generator=mask_generator, mask_times=mask_times,
                        summarize_img = summarize_img)
    
    dataloader = DataLoader(input_data, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return dataloader

def get_dino_dataloader(input_path,  batch_size=8, n_months = 5, n_bands = 5, img_size = 96, patch_size = 3, tile_patches = None):
    
    
    mask_generator = MaskingGenerator(
            input_size=(img_size // patch_size, img_size // patch_size),
            max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
        )
    
    ntokens = (img_size//patch_size)**2
    m = partial(collate_data_and_cast, mask_ratio_tuple = [0.1,0.5], 
        mask_probability=0.5, 
        n_tokens= ntokens, 
         mask_generator= mask_generator,
        dtype= torch.float,)
    
    input_data = DinoSatTileDataset(nc_path=input_path, n_months=n_months, img_size = img_size, 
                            n_bands = n_bands, tile_patches = tile_patches)
    
    dataloader = DataLoader(input_data, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=m)
    
    return dataloader