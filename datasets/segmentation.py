import rioxarray as rio
import numpy as np

class TargetSegmentation():
    
    @property
    def xrdata(self):
        return self._xrdata
    
    def __init__(self, list_files = None, pixel_size = 96):
        self._list_files = list_files
        if self._list_files is not None:
            self._list_files.sort()
        self._xrdata = None
        self.pixel_size = pixel_size
    
    def read_xrarray(self, id_file = None, filename = None):
        self._tmp_file_path = filename if id_file is None else self._list_files[id_file]
        
        if self._tmp_file_path.endswith('.tif'):
            self._xrdata = rio.open_rasterio(self._tmp_file_path)
        else:
            raise ValueError("Unsupported files format, 'tif' files are only supported")
    
    def transform_to_numpy(self, invert_yaxis = False):
        values = self.xrdata.values[0] ## images are iverted on y axis
        
        if values.shape[1] < self.pixel_size or values.shape[0] < self.pixel_size:
            imgpadded= np.zeros([self.pixel_size,self.pixel_size]).astype(values.dtype)
            imgpadded[:values.shape[0],:values.shape[1]] = values
            values = imgpadded
        
        if invert_yaxis:
            return values[::-1]
        else:
            return values
    
    def get_target_image(self, id_file = None, filename = None, invert_yaxis=False):
        self._xrdata = None
        self.read_xrarray(id_file=id_file, filename=filename)
    
        return self.transform_to_numpy(invert_yaxis)