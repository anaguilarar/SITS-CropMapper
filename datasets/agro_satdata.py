

import math
import random
import os
import warnings
from datetime import datetime
from typing import List, Optional, Tuple, Union
import glob
import numpy as np
import xarray
import itertools
from tqdm import tqdm

def summarize_data_by_timegroups(data, dates, days,total_months = 5, reference_date = None):
    
    times_incycle = math.ceil((total_months*30)/days)
    
    grouped_values = np.zeros([times_incycle] + list(data.shape[1:])).astype(data.dtype)
    

    last_date =  np.copy(reference_date) if reference_date is not None else dates[-1]
    
    groups = groupdates_by_ndays(dates, days, times_incycle, ref_bwdate = np.copy(last_date) )
    
    refdate = None
    avgdates = []
    for i in range(times_incycle):
        
        sel = np.array(groups) == i
        if np.any(sel):
            avgdates.append(np.mean(dates[sel].astype(float)).astype(dates.dtype))
            #warnings.simplefilter("ignore", category=RuntimeWarning)
            grouped_values[i] = np.nanmean(data[sel], axis = 0)
        else:
            refdate=last_date-(np.timedelta64(days*(times_incycle-(i+1)),'D'))
            avgdates.append(refdate)

    grouped_values[np.isnan(grouped_values)] = 0
    return grouped_values, avgdates

def groupdates_by_ndays(dates, days, times_incycle, ref_bwdate = None ):

    """Starting from last date

    Returns:
        _type_: _description_
    """

    #refdate=dates[-1]
    refdate = ref_bwdate if ref_bwdate else dates[-1]
    groups = np.zeros(dates.shape[0])

    for i in range((times_incycle-1),-1, -1):
        sel = np.where(np.logical_and(dates <= (refdate ), dates >(refdate - np.timedelta64(days,'D'))))[0]
        if len(sel)==1:
            groups[sel[0]] = i
        elif len(sel)>1:
            sel.sort()
            groups[sel[0]:(sel[-1]+1)] = i# * ((sel[-1] - sel[0])+1)
            
        refdate-=np.timedelta64(days,'D')

    return list(groups[:len(dates)])

def getting_datesdiff_in_days(date1:np.datetime64, date2:np.datetime64):    
    return ( date2 - date1).astype('timedelta64[D]') / np.timedelta64(1, 'D')

def first_day_of_month(date):
    startingmonth = date.astype('datetime64[M]').astype(int) %12 +1
    startingyear = date.astype('datetime64[Y]').astype(int) + 1970
    startingmonth = f'0{startingmonth}' if startingmonth<10 else startingmonth
    
    return np.datetime64(datetime.strptime(f'{startingyear}{startingmonth}01', '%Y%m%d'))

def dates_to_continousdays(dates:np.datetime64, reference_date = None) -> np.array:
    dates = np.sort(dates)
    #init_date = first_day_of_month(dates[0])
    if reference_date is not None:
        datesc = ((reference_date-dates).astype('timedelta64[D]') / np.timedelta64(1, 'D'))
    else:
        datesc = ((dates[-1]-dates).astype('timedelta64[D]') / np.timedelta64(1, 'D'))

    return np.abs(datesc - datesc[0])

def adding_dates(img_values: np.ndarray, days_list: List):
    """
    img_values must has dimensions T x C x H x W
    """
    days_list = sorted(days_list)
    if len(img_values.shape) != 4: return None
    daysdata = np.zeros([img_values.shape[0]]+[1]+list(img_values.shape[2:])).astype(img_values.dtype)
    daysdata[:,0,0,0] = days_list

    return np.concatenate([img_values, daysdata], axis = 1)
        

def dates_padding(img_values, nmax_dates = 100):
    """

    Args:
        img_values (_type_): T x C x H x W
        nmax_dates (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """
    daysdata = np.zeros([nmax_dates-img_values.shape[0]]+[img_values.shape[1]]+list(img_values.shape[2:])).astype(img_values.dtype)
    return np.concatenate([img_values, daysdata], axis = 0)     

class GrowingSason_SatData():
    """A class for handling and processing satellite data over a growing season.
    
    This class provides methods to read, process, and visualize satellite data
    stored in NetCDF format over a specified growing season window.

    Parameters
    ----------
    list_files : Optional[List[str]], optional
        List of file paths containing satellite data, by default None
    months_season : int, optional
        Number of months to consider as a growing season, by default 5
    pixel_size : int, optional
        Expected size of the image pixels, by default 48
    """

    def __init__(self, list_files: Optional[List[str]] = None, 
                 months_season: int = 5, 
                 pixel_size: int = 48):
        """Initialize the GrowingSeasonSatData instance."""
        
        self._date_colname:str = 'date'
        self._x_colname:str = 'x'
        self._y_colname:str = 'y'
        self._list_files = list_files
        if self._list_files is not None: self._list_files.sort()
        self.n_months = months_season
        self._xrdata = None
        self.pixel_size = pixel_size
    
    def read_xrdataset(self, 
                    id_file: Optional[int] = None, 
                    engine: str = 'netcdf4', 
                    chunk_size: Optional[int] = None, 
                    filename: Optional[str] = None) -> None:
        """Read satellite data from a NetCDF file into an xarray Dataset.

        Parameters
        ----------
        id_file : Optional[int], optional
            Index of file in list_files to read, by default None
        engine : str, optional
            Engine to use for reading NetCDF file, by default 'netcdf4'
        chunk_size : Optional[int], optional
            Chunk size for dask arrays, by default None
        filename : Optional[str], optional
            Direct file path to read, by default None

        Raises
        ------
        ValueError
            If file format is not supported (only .nc files supported)
        """
        fn = filename if id_file is None else self._list_files[id_file]
        self._tmp_file_path = fn
        if self._tmp_file_path.endswith('.nc'):
            with xarray.open_dataset(
                self._tmp_file_path, engine=engine, chunks=chunk_size, mask_and_scale=False
            ) as ds:
                return ds.copy()
        else:
            raise ValueError("Unsupported file format. Use '.nc'")
    
    def randomly_time_window_selection(self, 
                                    starting_month: Optional[int] = None, 
                                    ending_month: Optional[int] = None) -> None:
        """Randomly select a time window of n months from the dataset.

        Parameters
        ----------
        starting_month : Optional[int], optional
            Earliest possible starting month (1-12), by default None
        ending_month : Optional[int], optional
            Latest possible ending month (1-12), by default None
        """
        
        dates = self._xrdata[self._date_colname].values
        months = [i.astype('datetime64[M]').astype(int)%12+1 for i in dates]

        minmonthup = ending_month - self.n_months if ending_month is not None else (np.max(months) - (self.n_months-1))
        minmonthdown = starting_month or np.min(months)
        
        if minmonthup<minmonthdown: 
            minmonthup = minmonthdown

        starting = random.randint(minmonthdown,minmonthup)
        dates_toselect = [i for i, m in enumerate(months) if m in range(starting,starting+(self.n_months))]
        self._xrdata = self._xrdata.isel({self._date_colname: dates_toselect})
        
    def time_window_selection(self, 
                            starting_date: Union[str, np.datetime64] = None, 
                            ending_date: Optional[Union[str, np.datetime64]] = None) -> None:
        """Select a specific time window from the dataset.

        Parameters
        ----------
        starting_date : Union[str, np.datetime64], optional
            Starting date of the window (string or np.datetime64)
        ending_date : Optional[Union[str, np.datetime64]], optional
            Ending date of the window, by default None (will use n_months*30 days)
        """
        if starting_date is None:
            if isinstance(ending_date, str): ending_date = np.array(ending_date).astype('datetime64[D]')
            starting_date = ending_date  - np.array((self.n_months*30)+1, 'timedelta64[D]')
        else:
            if isinstance(starting_date, str): starting_date = np.array(starting_date).astype('datetime64[D]')
            ending_date = ending_date or starting_date + np.array((self.n_months*30)+1, 'timedelta64[D]')
        self._xrdata = self._xrdata.sel({self._date_colname : np.logical_and(starting_date<= self._xrdata[self._date_colname].values, 
                                                                            ending_date>= self._xrdata[self._date_colname].values)})

    def transform_to_numpy(self, 
                        add_dates_aschannel: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, List[float]]]:
        """Convert the xarray Dataset to numpy array with optional date channel.

        Parameters
        ----------
        add_dates_aschannel : bool, optional
            Whether to add normalized dates as an additional channel, by default True

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, List[float]]]
            Either the array with date channel or tuple of (array, dates)
        """
        days_with_data = dates_to_continousdays(self._xrdata[self._date_colname].values)
        days_with_data = np.array(days_with_data) / (self.n_months*30)
        if self._xrdata is None: return None    
        trvalues = self._xrdata.to_array().values.swapaxes(0,1) # T x C x H x W
        trvalues[np.isnan(trvalues)] = 0
        if add_dates_aschannel: 
            return dates_padding(adding_dates(trvalues, days_with_data.tolist()))
        else:
            return trvalues, days_with_data.tolist()

    def check_axis_direction(self):
        
        if (self._xrdata.y.values[1]-self._xrdata.y.values[0]) > 0:
            self._invertyaxis = True
        else:
            self._invertyaxis = False
    
    def summarize_xrdata_by_timewindow(self, days_interval: int = 14, reference_date = None) -> np.ndarray:
        """Convert xarray data to numpy array with temporal summarization.

        Parameters
        ----------
        days_interval : int, optional
            Number of days to group for temporal summarization, by default 14
        reference_date: np.datetime
            use a reference date to calculate the last date
        Returns
        -------
        np.ndarray
            Processed numpy array with date information
        """
        trvalues = self.from_xarray_to_nparray()
        dates = self._xrdata[self._date_colname].values
        img_values, new_dates = summarize_data_by_timegroups(trvalues, dates, days_interval, total_months = self.n_months, reference_date = reference_date)
        self._new_dates = new_dates
        days_with_data = dates_to_continousdays(self._new_dates, reference_date = reference_date)
        days_with_data = np.array(days_with_data)# / (self.n_months*30)
        self.check_axis_direction()
        #return self.transform_to_numpy()
        return adding_dates(img_values, days_with_data.tolist())
    
    
    def get_satellite_as_image(self, 
                            id_file: Optional[int] = None, 
                            filename: Optional[str] = None, 
                            days_interval: int = 14,
                            reference_date = None) -> Optional[np.ndarray]:
        """Get satellite data as processed numpy array image.

        Parameters
        ----------
        id_file : Optional[int], optional
            Index of file to read, by default None
        filename : Optional[str], optional
            Direct file path to read, by default None
        days_interval : int, optional
            Days interval for temporal summarization, by default 14

        Returns
        -------
        Optional[np.ndarray]
            Processed image array or None if no data
        """
        self._xrdata = self.read_xrdataset(id_file=id_file, filename=filename)
        
        self.randomly_time_window_selection()
        if self._xrdata is None: return None 
        return self.summarize_xrdata_by_timewindow(days_interval, reference_date= reference_date)

        
    def from_xarray_to_nparray(self) -> np.ndarray:
        """Preprocess the numpy array from xarray data.

        Returns
        -------
        np.ndarray
            Preprocessed array with proper scaling and padding
        """
        _SPECTRAL_BANDS = ['blue', 'green', 'red', 'nir', 'swir1', 'ndvi', 'gndvi']
        band_vars = [v for v in _SPECTRAL_BANDS if v in self._xrdata.data_vars]
        trvalues = self._xrdata[band_vars].to_array().values.swapaxes(0,1)
        trvalues[np.logical_and(trvalues>1, trvalues< 100000)] = trvalues[np.logical_and(trvalues>1, trvalues< 100000)]*0.0001
        trvalues[trvalues<-10] = np.nan
        if trvalues.shape[2] == self.pixel_size and trvalues.shape[3] == self.pixel_size: return trvalues
        ## image padding
        imgpadded= np.zeros([trvalues.shape[0], trvalues.shape[1] , self.pixel_size,self.pixel_size]).astype(trvalues.dtype)
        imgpadded[:,:,:trvalues.shape[2],:trvalues.shape[3]] = trvalues
        return imgpadded

        
    def plot_images(self, 
                id_image: Optional[int] = None, 
                ndarray_img: Optional[np.ndarray] = None, 
                scale_factor: float = 5,
                labels = None,
                **kwargs):
        """Plot satellite images with RGB channels.

        Parameters
        ----------
        id_image : Optional[int], optional
            Index of image to plot, by default None
        ndarray_img : Optional[np.ndarray], optional
            Multitemporal array with dimensions T x C x H x W
        scale_factor : float, optional
            Scaling factor for visualization, by default 5
        **kwargs
            Additional arguments passed to plot_multichanels

        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        from . import plot_multichanels
        
        if id_image is not None:
            ndarray_img = self.get_satellite_as_image(id_image = id_image)
        
        if ndarray_img is not None:
            labels = labels or (ndarray_img[:,-1,0,0]*(self.n_months*30 +.0001)).astype(int)
            f = plot_multichanels(ndarray_img[:,[2,1,0]].swapaxes(1,2).swapaxes(2,3)*scale_factor, 
                    chanels_names = labels, **kwargs)
            
        else:
            raise ValueError('please provide a valid data or id image')
        
        
class MltTileData(GrowingSason_SatData):
    def __init__(self, path, months_season = 7, pixel_size = 48, tile_patches = None):
        
        self.path = path
        
        self.filesinpath = [i for i in os.listdir(self.path) if i.endswith('.nc')]
        
        self._patch_str = 'patch_'
        self._patch_ids = None
        self._tiles_ids = None
        self._all_combinations= tile_patches
        
        super().__init__(months_season=months_season, pixel_size=pixel_size)
    
    @property
    def patch_ids(self):
        if self._patch_ids is None:
            uniquepatch = list(set(i[i.index(self._patch_str)+len(self._patch_str):-3] for i in self.filesinpath))
            uniquepatch.sort()
            self._patch_ids = uniquepatch
        return self._patch_ids
    
    @property
    def tiles_ids(self):
        if self._tiles_ids is None:
            uniquetiles = list(set(i[:(i.index(self._patch_str)-6)] for i in self.filesinpath))
            uniquetiles.sort()
            self._tiles_ids = uniquetiles
        return self._tiles_ids
    
    @property
    def all_combinations(self):
        if self._all_combinations is None:
            uniquecombs = list(itertools.product(self.tiles_ids, self.patch_ids))
            comb_withdata = []
            for i in tqdm(range(len(uniquecombs))):
                values = self.find_mlt_path(*uniquecombs[i])
                if len(values)>0:
                    comb_withdata.append(i)
                    
            self._all_combinations = [uniquecombs[i] for i in comb_withdata]
        return self._all_combinations

    def find_mlt_path(self, tile_id:str, patch_id:str):
        """Retrieve multitemporal files"""
        return glob.glob(self.path + f'/{tile_id}_*patch_{patch_id}.nc')
    
    
    def get_tiles_data_per_patch(self, idx):
        patch_id = 'patch_{}'.format(self.patch_ids[idx])
        return self.get_tiles_data(patch_id)
    
    def get_tiles_data(self, idx:int):

        mlt_patch_filenames = self.find_mlt_path(*self.all_combinations[idx])
        if len(mlt_patch_filenames)==0: return None
        datalist = [self.read_xrdataset(filename=filepath) for filepath in mlt_patch_filenames]
        self._xrdata = xarray.concat(datalist, dim = self._date_colname)
        return self._xrdata
    
    def get_random_date_tile_as_image(self, 
                            idx: Optional[int] = None, 
                            days_interval: int = 14) -> Optional[np.ndarray]:
        """Get satellite data as processed numpy array image.

        Parameters
        ----------
        id_file : Optional[int], optional
            Index of file to read, by default None
        filename : Optional[str], optional
            Direct file path to read, by default None
        days_interval : int, optional
            Days interval for temporal summarization, by default 14

        Returns
        -------
        Optional[np.ndarray]
            Processed image array or None if no data
        """
        self.get_tiles_data(idx)
        dates = self._xrdata.date.values
        upto = self._xrdata.date.values[-1] - np.array(((self.n_months-2)*30)+1, 'timedelta64[D]')
        
        starting_date = random.choice(dates[dates<=upto])
        self.time_window_selection(starting_date=starting_date)

        if self._xrdata is None: return None 
        imgvals = self.summarize_xrdata_by_timewindow(days_interval)
        self._xrdata = None
        return imgvals
        
    

