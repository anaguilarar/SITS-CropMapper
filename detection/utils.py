import torch
import xarray
from tqdm import tqdm
import numpy as np

DAYS_IN_MONTH = {
    1: 31,
    2: 28,  # 29 in a leap year
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31
}

def cover_prediction(model, x, x2 = None):
    if x2 is not None:
        outimg = model.predict(x.unsqueeze(0), x2.unsqueeze(0))
    else:
        outimg = model.predict(x.unsqueeze(0))
    predicted = torch.argmax(outimg.data[0], 0).to('cpu').detach().numpy()

    return predicted

def predict_tile(tile_id, model, dataset, date, summary_layer = True):
    
    if summary_layer:
        img, img2 = dataset.__getitem__(tile_id, starting_date=None, ending_date = date, reference_date = np.array(date).astype('datetime64[D]'))
        
        prediction = cover_prediction(model, img, img2)
    else:
        img = dataset.__getitem__(tile_id, starting_date=None, ending_date = date, reference_date = np.array(date).astype('datetime64[D]'))
        prediction = cover_prediction(model, img)
        
    sp_data = xarray.zeros_like(dataset._xrdata.isel(date = 0).blue)
    sp_data.values = prediction
    sp_data = sp_data.rename('prediction')
    sp_data.attrs = dict(dataset._xrdata.attrs)
    sp_data.attrs['long_name'] = 'prediction'
    return sp_data
    
def predict_over_an_area(model, dataset, date, summary_layer = None):
    
    tiles_prediction = []
    for idx in tqdm(range(dataset.__len__())):
        try:
            sp_data = predict_tile(idx, model, dataset, date, summary_layer)
            tiles_prediction.append(sp_data.drop('date'))        
        except:
            continue
        
    return tiles_prediction