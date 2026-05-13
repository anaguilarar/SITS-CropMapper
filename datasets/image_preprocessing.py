import warnings

import numpy as np
from statsmodels.nonparametric.kernel_regression import KernelReg

from scipy.signal import savgol_coeffs
from scipy.ndimage import convolve1d


def chen_sg_filter(curve_0, max_iteration=10, coeffs_trend1 = None, coeffs_trend2 = None):
    '''
    Taken from https://github.com/LicongLiu/SMF_S_Release/blob/main/src/chensg.py
    A simplified version of the Chen SG filter for vegetation index timeseries smooth.
    Please refer to the article for details:
        A simple method for reconstructing a high-quality NDVI time-series data set based on the Savitzky-Golay filter.
    Note:
        This algorithm is just an simplified python implementation of that article, just to demonstrate how to use the
        SMFS algorithm. The complete Chen SG filtering algorithm can be downloaded from this website:
        http://www.chen-lab.club/?post_type=products&page_id=14968
    :return:
    '''
    if coeffs_trend1 is None:
        coeffs_trend1 = savgol_coeffs(7, 5)
    if coeffs_trend2 is None:
        coeffs_trend2 = savgol_coeffs(3, 2)
    curve_tr = convolve1d(curve_0, coeffs_trend1, mode="wrap")
    d = curve_tr - curve_0
    dmax = np.max(np.abs(d))
    w_func = np.frompyfunc(lambda d_i: min((1, 1 - d_i/dmax)), 1, 1)
    W = w_func(d)
    curve_k = np.copy(curve_tr)
    f_arr = np.zeros(max_iteration)
    curve_previous = None
    for i in range(max_iteration):
        curve_k = np.maximum(curve_k, curve_0)
        curve_k = convolve1d(curve_k, coeffs_trend2, mode="wrap")
        f_arr[i] = np.sum(np.abs(curve_k - curve_0) * W)
        if i >= 1 and f_arr[i] > f_arr[i - 1]:
            return curve_previous
        curve_previous = curve_k
    return curve_previous


def kernel_regression(y, bw, int_days):
    y = np.asarray(y).flatten()

    # Handle all-NaN case immediately
    if np.all(np.isnan(y)):
        warnings.warn("kernel_regression: all-NaN input — returning NaN array.")
        return np.full_like(y, np.nan)

    # Convert dates to day offsets
    
    x = int_days.astype(float)
    
    # Clean NaNs
    valid_mask = ~np.isnan(y)
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    # Check sufficient data
    if len(y_clean) < 3:  # Need at least 3 points for regression
        warnings.warn(f"kernel_regression: only {len(y_clean)} valid points — falling back to NaN-fill.")
        result = np.full(len(y), np.nan)
        result[valid_mask] = y_clean
        return result

    # Scale coordinates
    x_min, x_max = x_clean.min(), x_clean.max()
    x_range = max(x_max - x_min, 1e-6)  # Prevent division by zero
    x_scaled_clean = (x_clean - x_min) / x_range

    # Fit model with stability checks
    try:
        model = KernelReg(
            endog=y_clean,
            exog=[x_scaled_clean],
            var_type='c',
            reg_type='lc',
            bw=[bw]
        )
        # Predict at original time points
        x_scaled_full = (x - x_min) / x_range
        mean, _ = model.fit(x_scaled_full)
    except Exception as e:
        warnings.warn(f"kernel_regression: regression failed ({e}) — falling back to linear interpolation.")
        mean = np.interp(x, x_clean, y_clean)
    
    return mean
    