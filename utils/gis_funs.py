
import dask
import math
import numpy as np
import os
import pandas as pd

import rioxarray as rio
import rasterio
import re
import xarray

from shapely.geometry import Polygon
from dask.diagnostics import ProgressBar
from itertools import product
from rasterio import windows
from rasterio.transform import Affine
from tqdm import tqdm
from .general import VEGETATION_INDEX


def clip_xarray(xrdata: xarray.DataArray, geom_feat):
    src_crs = xrdata.spatial_ref.crs_wkt
    geom_proj = geom_feat.to_crs(src_crs)
    return xrdata.rio.clip(geom_proj.geometry.values, geom_proj.crs, all_touched = True)


# Define function to scale 
def scaling(band):
    scale_factor = band.attrs['scale_factor'] 
    band_out = band.copy()
    band_out.data = band.data*scale_factor
    band_out.attrs['scale_factor'] = 1
    return(band_out)


def crop_using_windowslice(xrdata: xarray.Dataset, 
                           window: windows.Window, transform: Affine) -> xarray.Dataset:
    """
    Crop an xarray dataset using a window.

    Parameters:
    -----------
    xr_data : xr.Dataset
        The xarray dataset to be cropped.
    window : Window
        The window object defining the cropping area.
    transform : Affine
        Affine transformation defining the spatial characteristics of the cropped area.

    Returns:
    --------
    xr.Dataset
        Cropped xarray dataset.
    """
    
    # Extract the data using the window slices
    
    xrwindowsel = xrdata.isel(y=window.toslices()[0],
                                     x=window.toslices()[1]).copy()

    xrwindowsel.attrs['width'] = xrwindowsel.sizes['x']
    xrwindowsel.attrs['height'] = xrwindowsel.sizes['y']
    xrwindowsel.attrs['transform'] = transform

    return xrwindowsel

# adapated from https://github.com/Devyanshu/image-split-with-overlap
def start_points(size, split_size: int, overlap:float=0.0):
    """
    Calculate start points for tiling a dimension.

    Given a total size of the dimension, tile size, and overlap between tiles,
    this function calculates and returns the start points for each tile,
    ensuring coverage of the entire dimension.

    Parameters
    ----------
    size : int
        The total size of the dimension to be tiled.
    split_size : int
        The size of each tile.
    overlap : float, optional
        The fraction of overlap between consecutive tiles, by default 0.0.

    Returns
    -------
    List[int]
        A list of start points for each tile.
    """
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(pt)
            break
        else:
            points.append(pt)
        counter += 1
    return points

def get_tiles(ds, nrows=None, ncols=None, width=None, height=None, overlap=0.0):
    """

    :param ds: raster metadata
    :param nrows:
    :param ncols:
    :param width:
    :param height:
    :param overlap: [0.0 - 1]
    :return:
    """
    # get width and height from xarray attributes
    ncols_img, nrows_img = len(ds.x.values), len(ds.y.values)

    if nrows is not None and ncols is not None:
        width = math.ceil(ncols_img / ncols)
        height = math.ceil(nrows_img / nrows)

    # offsets = product(range(0, ncols_img, width), range(0, nrows_img, height))

    col_off_list = start_points(ncols_img, width, overlap)
    row_off_list = start_points(nrows_img, height, overlap)

    offsets = product(col_off_list, row_off_list)
    big_window = windows.Window(col_off=0, row_off=0, width=ncols_img, height=nrows_img)
    for col_off, row_off in offsets:
        window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.rio.transform())
        yield window, transform
        
def zscore(s, window, thresh=3, return_all=False):
    roll = s.rolling(window=window, min_periods=1, center=True)
    avg = roll.mean()
    std = roll.std(ddof=0)
    z = s.sub(avg).div(std)   
    m = z.between(-thresh, thresh)
    
    if return_all:
        return z, avg, std, m
    return s.where(m, avg)

def estimate_coverage(mlt_data):
    return [1-(np.isnan(mlt_data[i]).sum()/mlt_data[i].flatten().shape[0]) for i in range(mlt_data.shape[0])]

def check_crs_inxrdataset(xrdataset):
    if 'crs' in xrdataset.attrs.keys():
        crs = xrdataset.attrs['crs']
        if isinstance(crs, rasterio.crs.CRS):
            xrdataset.attrs['crs'] = crs.to_string()
    return xrdataset

def set_encoding(xrdata, compress_method = 'zlib'):
    return {k: {compress_method: True} for k in list(xrdata.data_vars.keys())}

def bands_from_expression(expression):
    symbolstoremove = ['*','-','+','/',')','.','(',' ','[',']']
    test = expression
    for c in symbolstoremove:
        test = test.replace(c, '-')
        

    test = re.sub('\d', '-', test)
    varnames = [i for i in np.unique(np.array(test.split('-'))) if i != '']
    
    return varnames


def create_quality_mask(quality_data, bit_nums: list = [1, 2, 3, 4, 5]):
    """
    Uses the Fmask layer and bit numbers to create a binary mask of good pixels.
    By default, bits 1-5 are used.
    """
    quality_data = np.copy(quality_data)
    mask_array = np.zeros((quality_data.shape[0],quality_data.shape[1], quality_data.shape[2]))
    #mask_array = np.zeros((quality_data.shape[0], quality_data.shape[1]))
    # Remove/Mask Fill Values and Convert to Integer
    quality_data = np.nan_to_num(quality_data, 0).astype(np.int16)
    for bit in bit_nums:
        # Create a Single Binary Mask Layer
        mask_temp = np.array(quality_data) & 1 << bit > 0
        mask_array = np.logical_or(mask_array, mask_temp)
    return mask_array


def compute_vi_equation(xrdata, vi_equation):
    variable_names = list(xrdata.data_vars)
    varnames = bands_from_expression(vi_equation) 
    vi = xrdata[varnames[0]].copy()
    for i, varname in enumerate(varnames):
        if varname in variable_names:
            exp = (["xrdata['{}'].data".format(varname), varname])
            vi_equation = vi_equation.replace(exp[1], exp[0])
        else:
            raise ValueError('there is not a variable named as {}'.format(varname))
    vi_data = eval(vi_equation)
    vi.data = vi_data
    # exclude the inf values
    vi = xarray.where(vi != np.inf, vi, np.nan, keep_attrs=True)
    
    return vi


def find_ts_anomaly(npdata, thresh:float = 2.5, window= 7,):
        #npdata = xrarray.to_numpy()
        #
        assert len(npdata.shape) ==3, "Data must have dimensions T x W x H"
        ts_data = np.nanmean(npdata.reshape((npdata.shape[0],npdata.shape[1]*npdata.shape[2])), axis= 1)

        _, _, _, m = zscore(pd.DataFrame({'MW': ts_data})['MW'], window=window, thresh=thresh,return_all=True)

        return pd.DataFrame({'MW': ts_data}).loc[~m, 'MW'].index


def calculate_vi_fromxarray(xrdata, vi='ndvi', expression=None, label=None, overwrite = False):
    """
    Calculates vegetation indices from an xarray dataset.

    Parameters:
    ----------
    xrdata : xarray.Dataset
        The xarray dataset from which to calculate the vegetation index.
    vi : str, default 'ndvi'
        Name of the vegetation index to be calculated.
    expression : str, optional
        Custom expression for calculating the vegetation index.
    label : str, optional
        Label for the new vegetation index data in the dataset.
    overwrite : bool, default False
        If True, overwrite the existing data; otherwise, skip if already present.

    Returns:
    -------
    xarray.Dataset
        The xarray dataset with the new vegetation index data added.
    """
    
    if expression is None and vi in list(VEGETATION_INDEX.keys()):
        expression = VEGETATION_INDEX[vi]

    vi_layer = compute_vi_equation(xrdata, expression)
    # change the long_name in the attributes
    vi_layer.attrs['long_name'] = vi
    vi_layer.attrs['scale_factor'] = 1
    xrdata_vi = xarray.merge([xrdata, vi_layer.to_dataset(name = vi)])
    
    return xrdata_vi


def split_xrdataset_into_tiles(xrdata, output_path, patch_size, bands = None):
    
    if not os.path.exists(output_path): os.mkdir(output_path)
    # https://dask.discourse.group/t/how-to-efficiently-extract-patches-from-a-xarray-dask-dataset/2871
    patches = [list(range(len(xrdata.chunks['x']))) , list(range(len(xrdata.chunks['y'])))]
    tasks = []
    for patch_idx in tqdm(patches[0], total=len(patches[0]), desc="Create patches across the X axis"):
        i= patch_idx* patch_size
        for patch_idy in tqdm(patches[1], total=len(patches[1]), desc="Create patches across the y axis"):
            j= patch_idy* patch_size
            filename = os.path.join(output_path, f"tile_{patch_idx}_{patch_idy}.nc")
            if not os.path.exists(filename):
                patch = xrdata.isel(x=slice(i, min(i + patch_size, xrdata.x.size)), y=slice(j, min(j + patch_size, xrdata.y.size)))
                delayed_obj = patch[bands].astype(float).to_netcdf(filename, format="NETCDF4", engine="netcdf4", compute=False)
                tasks.append(delayed_obj)  
    if len(tasks)>0:
        print("Saving patches")
        with ProgressBar():
            dask.compute(*tasks)
            

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
