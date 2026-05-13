import numpy as np

def perform(fun, *args):
    return fun(*args)

def perform_kwargs(fun, **kwargs):
    return fun(**kwargs)


def summarise_trasstring(values):
    
    if type (values) ==  list:
        paramsnames = '_'.join([str(j) 
        for j in values])
    else:
        paramsnames = values

    return '{}'.format(
            paramsnames
        )
    

def standard_scale(data: np.ndarray, meanval: float = None, stdval: float = None, navalue: float = 0) -> np.ndarray:
    """
    Standardizes the input data using the provided mean and standard deviation.

    Parameters
    ----------
    data : np.ndarray
        The data to be standardized.
    meanval : float, optional
        The mean value for standardization. Defaults to the mean of the data.
    stdval : float, optional
        The standard deviation value for standardization. Defaults to the standard deviation of the data.
    navalue : float, optional
        The value to be treated as NaN. Defaults to 0.

    Returns
    -------
    np.ndarray
        The standardized data.
    """
    if meanval is None:
        meanval = np.nanmean(data)
    if stdval is None:
        stdval = np.nanstd(data)
    if navalue == 0:
        datac1 = data.copy().astype(np.float64)
        ## mask na
        datac1[datac1 == navalue] = np.nan
        dasc = (datac1-meanval)/stdval 
        dasc[np.isnan(dasc)] = navalue 
    else:
        dasc= (data-meanval)/stdval

    return dasc


def minmax_scale(data: np.ndarray, minval: float = None, maxval: float = None, navalue: float = 0) -> np.ndarray:
    """
    Scales the input data using min-max normalization.

    Parameters
    ----------
    data : np.ndarray
        The data to be scaled.
    minval : float, optional
        The minimum value for scaling. Defaults to the minimum of the data.
    maxval : float, optional
        The maximum value for scaling. Defaults to the maximum of the data.
    navalue : float, optional
        The value to be treated as NaN. Defaults to 0.

    Returns
    -------
    np.ndarray
        The min-max scaled data.
    """
    
    if minval is None:
        minval = np.nanmin(data)
    if maxval is None:
        maxval = np.nanmax(data)
    
    if navalue == 0:
        ## mask na
        data[data == navalue] = np.nan
        dasc = (data - minval) / ((maxval - minval)) 
        dasc[np.isnan(dasc)] = navalue 
    else:
        dasc= (data - minval) / ((maxval - minval))
    
    return dasc