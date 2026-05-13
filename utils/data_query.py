import concurrent.futures
import dask
import earthaccess
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import rioxarray as rio
import rasterio
import shutil
import xarray

from abc import ABC, abstractmethod
from datetime import datetime
from dask.diagnostics import ProgressBar
from tqdm import tqdm

from .gis_funs import scaling, clip_xarray
from datasets.sat_preprocessing import Satellite_preprocessing

HLS_CNAMES= {
    'L30':{
        'B02': 'blue',
        'B03': 'green',
        'B04': 'red',
        'B05': 'nir',
        'B06': 'swir01',
        'B07': 'swir02',
        'B09': 'cirrus',
        'Fmask' : 'Fmask'
    },
    'S30': {
        'B02': 'blue',
        'B03': 'green',
        'B04': 'red',
        'B05': 'edge1',
        'B06': 'edge2',
        'B07': 'edge3',
        'B08': 'nirbroad',
        'B8A': 'nir',
        'B11': 'swir01',
        'B12': 'swir02',
        'B10': 'cirrus',
        'Fmask' : 'Fmask'
    }
}

            
class DownloadSatelliteBase(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def query_data(self):
        pass
    
    @abstractmethod
    def download_data(self):
        pass
    
    


def download_hls_sat_product(url, chunk_size, maxtries = 3, roi_windows = None):
    """
    the connection can be lost, so it is worth to try a couple of times

    Returns:
        _type_: _description_
    """
    pr = None
    gdal_config = {'GDAL_HTTP_COOKIEFILE':'~/cookies.txt',
                        'GDAL_HTTP_COOKIEJAR': '~/cookies.txt',
                        'GDAL_DISABLE_READDIR_ON_OPEN':'EMPTY_DIR',
                        'CPL_VSIL_CURL_ALLOWED_EXTENSIONS':'TIF',
                        'GDAL_HTTP_UNSAFESSL': 'YES',
                        'GDAL_HTTP_MAX_RETRY': '10',
                        'GDAL_HTTP_RETRY_DELAY': '0.5'}
    
    for i in range(maxtries):
        try:
            with rasterio.Env(**gdal_config):
                pr = rio.open_rasterio(url, chunks=chunk_size, masked= True).squeeze('band', drop=True)
            break
        except:
            pass
    
    return pr


def create_dimension(xrdata_dict, newdim_name = 'date', isdate = True):
    datacube_mrs = []
    for k,v in tqdm(xrdata_dict.items()):
        xrtemp = v.expand_dims(dim = [newdim_name])
        xrtemp[newdim_name] = [k]
        datacube_mrs.append(xrtemp)
        
    datacube_mrs = xarray.concat(datacube_mrs, dim = newdim_name)
    if isdate:
        datacube_mrs[newdim_name] = [datetime.strptime(i, "%Y%m%d") for i in list(xrdata_dict.keys())]
    return datacube_mrs

class HLSDATA(DownloadSatelliteBase):
    @property
    def product(self):
        return 'hls'
    
    def __init__(self, chunk_size = 96):
        self.products = None
        self._qresults = None
        self._set_up()
        self._chunk_size = chunk_size
        
    
    def _set_up(self):
        earthaccess.login(persist=True)

    @property
    def dates(self):
        df = pd.json_normalize(self._qresults)
        return df['umm.TemporalExtent.RangeDateTime.BeginningDateTime'].apply(lambda x: 
            datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ")).values
    
    @property
    def url_links(self):
        self._urls    
    
    @property
    def cloud_coverage(self):
        cc = None
        if self._qresults is not None:
            cc = pd.json_normalize(self._qresults)['umm.AdditionalAttributes'].apply(lambda x: float(x[1]['Values'][0]))
                    
        return cc.values

    def query_data(self, geom_feature:gpd.GeoDataFrame, period, max_cloud_coverage:int =None, max_count = 100):
        
        self._geom_feature = geom_feature
        bbox = tuple(list(geom_feature.total_bounds))
        
        self._qresults = earthaccess.search_data(
            short_name = ['HLSL30', 'HLSS30'],
            bounding_box = bbox,
            temporal = period,
            count = max_count
        )
        
        if max_cloud_coverage:
            self._qresults = [self._qresults[i] for i in np.where(self.cloud_coverage<max_cloud_coverage)[0]]
        
        self._urls = [granule.data_links() for granule in self._qresults]
        
        return self._urls

    def download_data(self, url, channels):
        
        chunk_size = dict(band=1, x=self._chunk_size, y=self._chunk_size)
        channel_list = []
        
        rname = HLS_CNAMES['L30'] if url[0].rsplit('/')[-3] == 'HLSL30.020' else HLS_CNAMES['S30']
        rchannels = {k:v for k,v in rname.items() if v in channels}
        
        for u in url:
            u_channel = u.rsplit('.',-2)[-2]
            
            if u_channel not in rchannels.keys(): continue
            
            pr = download_hls_sat_product(u,chunk_size=chunk_size)
            if self._geom_feature is not None:
                pr = clip_xarray(pr, self._geom_feature)
            
            if u_channel != 'Fmask':
                #bname = pr.attrs['long_name'].lower()
                pr.attrs['scale_factor'] = 0.0001#pr.attrs.get('scale_factor',0.0001)
                pr = scaling(pr)
                
            channel_list.append(pr.to_dataset(name = rchannels[u_channel]))
        
        
        return xarray.merge(channel_list)


    def download_and_save(self, id_utrl, channels, savetp_data = True, tmp_outputpath = 'tmp'):
            xrtemp = self.download_data(self._urls[id_utrl], channels)
            xrtemp = xrtemp.expand_dims(dim = ['time'])
            xrtemp= xrtemp.assign_coords({"time": [self.dates[id_utrl]]})
            
            strdate = np.datetime_as_string(self.dates[id_utrl], unit='D')#datetime.strftime(self.dates[d], '%Y%m%d')
            filename = f'data{id_utrl}_{strdate}.nc'
            if savetp_data:
                Satellite_preprocessing().save_asnc(
                    xrdata = xrtemp.astype(float),fn = os.path.join(tmp_outputpath,filename), compute = True)
            return xrtemp
    
    def download_mlt_data(self, channels, ncores, tmp_outputpath = 'tmp', savetp_data = True):
        
        if os.path.exists(tmp_outputpath): shutil.rmtree(tmp_outputpath, ignore_errors=False, onerror=None)
        
        print(f'downloading hls data with {ncores} in {tmp_outputpath}')
        os.mkdir(tmp_outputpath)
        datacube_mrs = []
        if ncores >0:
            with tqdm(total=len(self._urls)) as pbar:
                with concurrent.futures.ProcessPoolExecutor(max_workers=ncores) as executor:
                    future_to_tr ={executor.submit(self.download_and_save, d, channels, savetp_data): (d) for d in range(len(self._urls))}
                    for future in concurrent.futures.as_completed(future_to_tr):
                        tr = future_to_tr[future]
                        try:
                            xrtemp = future.result()
                            datacube_mrs.append(xrtemp)
                        except Exception as exc:
                            print(f"Request for treatment {tr} generated an exception: {exc}")
                        pbar.update(1)
        else:
            for d in tqdm(range(len(self._urls))):
                try:
                    self.download_and_save(d, channels, savetp_data= savetp_data)
                except Exception as exc:
                    print(f"Request for {self._urls[d]} generated an exception: {exc}")

        if len(datacube_mrs)>0 and savetp_data:
            datacube_mrs = []  
            for datapath in os.listdir(tmp_outputpath):
                if datapath.endswith('.nc'): 
                    #chunk_size = dict(band=1, x=self._chunk_size, y=self._chunk_size)
                    datacube_mrs.append(xarray.open_dataset(os.path.join(tmp_outputpath, datapath), engine = "netcdf4"))

        datacube_mrs = xarray.concat(datacube_mrs, dim = 'time')
        datacube_mrs = datacube_mrs.sortby('time', ascending=True)
        datacube_mrs = datacube_mrs.transpose("time", "y", "x")
        #datacube_mrs['time'] = self.dates

        return datacube_mrs
        

"""
patches = patch_indices["Annotation"] + patch_indices["No-Annotation"]
tasks = []
for patch_idx in tqdm(patches, total=len(patches), desc="Creating patches"):
    i, j = patch_idx
    filename = Path(patch_folder, f"patch_{i}_{j}.nc")
    if not filename.exists():
        patch = S2.data.isel(x=slice(i, min(i + 128, S2.data.x.size)), y=slice(j, min(j + 128, S2.data.y.size)))
        delayed_obj = patch.to_netcdf(filename, format="NETCDF4", engine="netcdf4", compute=False)
        tasks.append(delayed_obj)  
if tasks:
    print("Saving patches")
    with ProgressBar():
        dask.compute(*tasks)
"""