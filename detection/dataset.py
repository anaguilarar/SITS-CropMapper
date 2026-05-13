
from datasets.agro_satdata import GrowingSason_SatData, MltTileData
from torch.utils.data import Dataset
import torch

import numpy as np
from datasets.transforms.tensor_transforms import Normalize, ToTensor


class InputImgDataset(Dataset, MltTileData):
    
    def __len__(self):
        return len(self.patch_ids)
    
    def __init__(self, nc_path, n_months = 5, n_bands = 5, img_size = 96, summarize_img = False):
        
        self._n_months = n_months
        self._img_size = img_size
        self.n_bands = n_bands
        self.summarize_img = summarize_img
        MltTileData.__init__(self,nc_path, months_season=self._n_months, pixel_size = self._img_size )
        self.normalize =  Normalize('hls', n_bands)
    
    def calculate_annual_summary(self, pixel_timeseries_np):
        # full_year_pixel_timeseries_np: (T_full_year, C_spectral) for one pixel

        valid_steps_mask = np.zeros_like(pixel_timeseries_np) # Mask for valid time steps
        valid_steps_mask[:] = pixel_timeseries_np
        valid_steps_mask[valid_steps_mask == 0] = np.nan

        median_summary = np.nanmedian(valid_steps_mask, axis = 0)
        median_summary[np.isnan(median_summary)] = 0# You could add other stats: np.median, np.std, np.percentile
            # summary_features = np.concatenate([mean_summary, median_summary, ...])
        return median_summary

    
    def __getitem__(self, idx, starting_date, ending_date = None, days_interval = 14, scale = True, reference_date = None):
        self.get_tiles_data(idx)
        dates = self._xrdata.date.values
        #starting_date = dates[starting_date]
        self.time_window_selection(starting_date=starting_date, ending_date = ending_date)
        
        satdata = self.summarize_xrdata_by_timewindow(days_interval, reference_date = reference_date)
        
        dates = satdata[:,-1]
        satdata = satdata[:,:-1]
        sat_image = ToTensor()(satdata)
        dates = torch.from_numpy(dates).unsqueeze(dim=1).to(torch.float32)
        
        sat_image = torch.concat([sat_image, dates], dim = 1)
        if scale:
            sat_image = Normalize('hls', self.n_bands)(sat_image)
        # dates
        
        if self.summarize_img:
            img_summarized = self.calculate_annual_summary(satdata)
            dummytime = np.zeros((1,img_summarized.shape[-2],img_summarized.shape[-1])).astype(img_summarized.dtype)
            img_summarized = np.concatenate([img_summarized, dummytime], axis = 0) # C, H, W
            img_summarized_t = ToTensor()(img_summarized)
            if scale:
                img_summarized_t = Normalize('hls', self.n_bands)(img_summarized_t.unsqueeze(dim=0)).squeeze()
            
            return sat_image, img_summarized_t[:-1]
        else:
            return sat_image#self.normalize(torch.concat([sat_image, dates], dim = 1))
        