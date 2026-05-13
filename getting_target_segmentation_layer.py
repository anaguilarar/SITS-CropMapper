from datasets.agro_satdata import GrowingSason_SatData
import os
import concurrent.futures
import rioxarray as rio
#import matplotlib.pyplot as plt
#from utils.gis_funs import clip_xarray
#from utils.gis_funs import get_xarray_polygon
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon
import xarray
## input

def clip_xarray(xrdata: xarray.DataArray, geom_feat):
    src_crs = xrdata.spatial_ref.crs_wkt
    geom_proj = geom_feat.to_crs(src_crs)
    return xrdata.rio.clip(geom_proj.geometry.values, src_crs, all_touched = True)


def from_xyxy_2polygon(x1: float, y1: float, x2: float, y2: float) -> Polygon:
    """
    Create a polygon from the coordinates of two opposite corners of a bounding box.

    Parameters:
    -----------
    x1 : float
        x-coordinate of the first corner.
    y1 : float
        y-coordinate of the first corner.
    x2 : float
        x-coordinate of the second corner.
    y2 : float
        y-coordinate of the second corner.

    Returns:
    --------
    shapely.geometry.Polygon
        Polygon geometry created from the bounding box coordinates.
    """
    
    xpol = [x1, x2,
            x2, x1,
            x1]
    ypol = [y1, y1,
            y2, y2,
            y1]

    return Polygon(list(zip(xpol, ypol)))

def get_xarray_polygon(xrdata: xarray.Dataset, dim1name: str = 'x', dim2name: str = 'y'):
    """
    Extracts the bounding polygon from an xarray.Dataset using specified dimension names.

    Parameters
    ----------
    xrdata : xarray.Dataset
        The input xarray dataset from which to extract the polygon.
    dim1name : str, optional
        The name of the first dimension, typically representing the x-axis, by default 'x'.
    dim2name : str, optional
        The name of the second dimension, typically representing the y-axis, by default 'y'.

    Returns
    -------
    Any
        The polygon object representing the bounding area of the dataset. The exact type
        depends on the implementation of `from_xyxy_2polygon`.

    Raises
    ------
    AssertionError
        If either `dim1name` or `dim2name` are not coordinates in `xrdata`.

    Notes
    -----

    """
    
    assert dim1name in list(xrdata.coords.keys()), f'{dim1name} is not in the xaray dimensions'
    assert dim2name in list(xrdata.coords.keys()), f'{dim2name} is not in the xaray dimensions'
    
    xcoords = xrdata.coords['x'].values
    ycoords = xrdata.coords['y'].values
    
    x1, x2 = np.min(xcoords), np.max(xcoords)
    y1, y2 = np.min(ycoords), np.max(ycoords)
    
    return from_xyxy_2polygon(x1, y1,
                              x2, y2)

def clip_target(idfile, files, target_path):
    target_data = rio.open_rasterio(target_path)
    dataset = GrowingSason_SatData(files, months_season= 6)
    dataset.read_xrdataset(idfile)
    if dataset._xrdata is not None:
        data_projected = dataset._xrdata.rio.reproject(target_data.rio.crs)

        polygonxr = get_xarray_polygon(data_projected)
        polygonxrdf = gpd.GeoDataFrame(index=[0], crs=target_data.rio.crs, geometry=[polygonxr]) 

        data = clip_xarray(target_data, polygonxrdf)

        if (dataset._xrdata.y.values[1]-dataset._xrdata.y.values[0]) > 0:
            if (data.y.values[1]-data.y.values[0])<0:
                data = data.assign_coords( y = data.y.values[::-1])

        filename = os.path.basename(dataset._tmp_file_path).split('.')[0]
        data.rio.to_raster(os.path.join('hls_data96/target_data',f'{filename}.tif'))

def main():

    nc_path = 'hls_data96/all_filtered'
    files = list(set(os.path.join(nc_path, i) for i in os.listdir(nc_path) if i.endswith('.nc')))
    files.sort()
    ## target
    
    target_path = 'data/reduced_class_.tif'
    
    ncores = 5
    
    #f = dataset.plot_images(id_image=10, num_rows= 3, num_columns=4,figsize=(16,8), invertaxis=False)
    with tqdm(total=len(files)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=ncores) as executor:
            future_to_tr ={executor.submit(clip_target, d, files, target_path): (d) for d in range(len(files))}
            for future in concurrent.futures.as_completed(future_to_tr):
                tr = future_to_tr[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"Request for treatment {tr} generated an exception: {exc}")
                pbar.update(1)
    #for d in tqdm(range(len(files))):
    #    clip_target(d, files, target_path)
        
        
    #    dataset._xrdata = None

if __name__ == '__main__':
    main()