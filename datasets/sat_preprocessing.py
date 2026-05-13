
import dask
import itertools
import numpy as np
import os
import xarray

from collections import Counter
from dask.diagnostics import ProgressBar
from datetime import datetime
from tqdm import tqdm
from typing import List, Optional, Union

from utils.gis_funs import (get_tiles, crop_using_windowslice, calculate_vi_fromxarray,clip_xarray, scaling, 
                        create_quality_mask, estimate_coverage,set_encoding, check_crs_inxrdataset,
                        find_ts_anomaly)


def save_patches(xrdata, output_path, patch_size, fn = None, bands = None):
    tiles = list(get_tiles(xrdata, width=patch_size, height=patch_size))
    n_patches = len(tiles)
    tasks = []
    fn = fn or 'patch'
    
    for pathch_id in tqdm(range(n_patches),total=n_patches, desc="Create patches"):
        tile_window,tr_aff = tiles[pathch_id]
        xrpatch = crop_using_windowslice(xrdata[bands], tile_window, tr_aff)
        filename = os.path.join(output_path, f"{fn}_{tile_window.col_off}_{tile_window.row_off}.nc")
        #if not os.path.exists(filename):
        delayed_obj = Satellite_preprocessing().save_asnc(
                xrdata = xrpatch.astype(float),fn = filename, compute = False)
            
            #delayed_obj = xrpatch.astype(float).to_netcdf(filename, format="NETCDF4", engine="netcdf4", compute=False)
        tasks.append(delayed_obj)  
    if len(tasks)>0:
        print("Saving patches")
        with ProgressBar():
            dask.compute(*tasks)


class Satellite_preprocessing():
    """
    A class for preprocessing satellite imagery data with a flexible pipeline execution.
    """
    
    def __init__(self, xrdata= None):
        """
        Initialize the class with an optional xarray dataset.

        Parameters
        ----------
        xrdata : Optional[xarray.Dataset], optional
            The input dataset, by default None.
        """
        self.xrdata = xrdata.copy() if xrdata is not None else None
    
    @staticmethod
    def compute_vi(xrdata: xarray.Dataset, vi: Union[str, List[str]]) -> xarray.Dataset:
        """
        Compute vegetation indices for the dataset.

        Parameters
        ----------
        xrdata : xarray.Dataset
            The input dataset.
        vi : str or list of str
            The vegetation indices to compute.

        Returns
        -------
        xarray.Dataset
            Dataset with computed vegetation indices.
        """
        vi = [vi] if not isinstance(vi, list) else vi
        for i in vi:
            xrdata = calculate_vi_fromxarray(xrdata, i) 
        return xrdata
    
    @staticmethod
    def clip(xrdata, geom_feature):
        return clip_xarray(xrdata, geom_feature)
    
    @staticmethod
    def re_scale(xrdata):
        return scaling(xrdata)
    
    @staticmethod
    def mask_using_fmask(xrdata: xarray.Dataset, quality_band: str = "Fmask", bit_nums = [1, 2, 3, 4, 5]) -> xarray.Dataset:
        """
        Apply a quality mask using quality mask layer.

        Parameters
        ----------
        xrdata : xarray.Dataset
            The input dataset.
        quality_band : str, optional
            Name of the quality band, by default "Fmask".

        Returns
        -------
        xarray.Dataset
            The dataset with applied mask.
        """
        
        quality_mask = xarray.apply_ufunc(create_quality_mask, xrdata[quality_band],
                                            output_core_dims= [["time"]], 
                                            input_core_dims= [["time"]], 
                                            #dask = 'allowed',
                                            #vectorize = True,
                                            )
        
        return xrdata.where(~quality_mask) 
    
    @staticmethod
    def group_by_single_date(xrdata: xarray.Dataset, time_coord_name: str = "time", time_coord_output = 'time') -> xarray.Dataset:
        """
        Group data by unique dates.

        Parameters
        ----------
        xrdata : xarray.Dataset
            The input dataset.
        time_coord_name : str, optional
            Name of the time coordinate, by default "time".

        Returns
        -------
        xarray.Dataset
            The dataset grouped by date.
        """
        dateformat = '%Y%m%d'
        xrdata.coords[time_coord_output] = xrdata[time_coord_name].dt.strftime(dateformat)
        xrdata = xrdata.groupby(time_coord_output).mean()
        dates =  [datetime.strptime(date, dateformat) for date in xrdata.date.values]
        xrdata = xrdata.assign_coords({time_coord_output: dates})
        return xrdata
    
    @staticmethod
    def filter_by_area_of_coverage(xrdata: xarray.Dataset, min_coverage: float, time_coord_name: str = "time"
    ) -> xarray.Dataset:
        """
        Filter data by minimum area coverage.

        Parameters
        ----------
        xrdata : xarray.Dataset
            The input dataset.
        min_coverage : float
            Minimum coverage percentage required.
        time_coord_name : str, optional
            Name of the time coordinate, by default "date".

        Returns
        -------
        xarray.Dataset
            Filtered dataset with sufficient coverage.
        """
        
        varname = list(xrdata.data_vars.keys())
        
        coverageareas = estimate_coverage(xrdata[varname[0]].to_numpy())
        times_tokeep = np.where(np.array(coverageareas)*100 > min_coverage)
        return xrdata.isel({time_coord_name:times_tokeep[0]})
    
        
    @staticmethod
    def remove_anomalies_from_ts(
        xrdata: xarray.Dataset, bands: Optional[List[str]] = None, thresh: float = 2.5, window: int = 9, time_coord_name: str = "time"
        ) -> xarray.Dataset:
        """
        Remove anomalies from time-series data.

        Parameters
        ----------
        xrdata : xarray.Dataset
            The input dataset.
        bands : list of str, optional
            Bands to check for anomalies, by default the first 3 bands.
        thresh : float, optional
            Z-score threshold for anomalies, by default 2.5.
        window : int, optional
            Rolling window size, by default 9.
        date_colname : str, optional
            Name of the date column, by default "date".

        Returns
        -------
        xarray.Dataset
            Dataset with anomalies removed.
        """
        
        bands = bands or list(xrdata.data_vars.keys())[:3]
        tp_to_remove = [find_ts_anomaly(xrdata[b].to_numpy(),thresh = thresh, window = window).values.tolist() for b in bands]
        tp = list(itertools.chain(*tp_to_remove))
        if tp:
            tp = [k for k,v in Counter(tp).items() if v>1]
            tp_to_keep = [i for i in range(xrdata[time_coord_name].shape[0]) if i not in np.unique(tp)]
            return xrdata.isel({time_coord_name: tp_to_keep})
        
        return xrdata
    
    @staticmethod
    def save_asnc(xrdata: xarray.Dataset, fn: str, compute = True) -> None:
        """
        Save a dataset to a NetCDF file with appropriate encoding.

        Parameters
        ----------
        xrdata : xarray.Dataset
            The dataset to save.
        fn : str
            Output file name.
        """
        dcengine = 'netcdf4'
        encoding = set_encoding(xrdata)
        xrdata = check_crs_inxrdataset(xrdata)
        return xrdata.to_netcdf(fn, encoding = encoding, engine = dcengine, compute = compute)
    
    @staticmethod
    def open_dataset(filepath: str, engine = 'netcdf4', chunk_size = None) -> xarray.Dataset:
        """
        Open a dataset from a NetCDF file.

        Parameters
        ----------
        filepath : str
            Path to the dataset file.

        Returns
        -------
        xarray.Dataset
        """
        if filepath.endswith('.nc'):
            with xarray.open_dataset(filepath, engine = engine, chunks  = chunk_size or "auto") as ds:
                return ds.copy()
        else:
            raise ValueError("Unsupported file format. Use '.nc'")
    @staticmethod
    def save_into_pathes(xrdata: xarray.Dataset, output_path: str, patch_size:int, fn:str = None, bands: Optional[List] = None) -> None:
        save_patches(xrdata, output_path, patch_size, fn, bands)
    
    def __call__(self, xrdata,pipeline_steps, params = None):
        params = params or {}
        xrp = xrdata.copy()
        for step in pipeline_steps:
            fun = getattr(self, step, None)
            if fun is not None:
                fun_params = params.get(step, {})
                xrp = fun(xrp, **fun_params)
                print(f'{step} completed')
            else:
                raise Warning(f'{step} skkiped beacasue it does not exist')
        
        return xrp