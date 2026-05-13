from typing import List, Optional, Union, Dict
from .utils import perform, perform_kwargs, summarise_trasstring

from .image_functions import (image_rotation,image_zoom,randomly_displace,
                              clahe_img, image_flip,shift_hsv,adding_noise,
                              illumination_shift, applying_denoise,
                              shear_image,cv2_array_type,perspective_transform)

import numpy as np
import cv2
import copy

import random
from PIL import Image


DEFAULT_PARAMS = {
                'rotation': random.randint(10,350),
                'zoom': random.choice([0.1,0.2,0.3, 0.4,0.5]),
                'clahe': random.randint(5,30),
                'shear': [random.choice(np.linspace(5,40,8)/100),
                          random.choice(np.linspace(5,40,8)/100)],
                'perspective': [random.choice(np.linspace(-10,10,10)/10000) ,
                                random.choice(np.linspace(-10,10,10)/10000) ],
                
                'shift': random.randint(5, 20),
                'flip': random.choice([-1,0,1]),
                'gaussian': random.choice([20,30,40, 50]),
                'hsv': [random.choice(list(range(-10,30,5))),
                        random.choice(list(range(-10,20,5))), 
                        random.choice(list(range(-10,20,5)))],
                'illumination':random.choice(list(range(-50,50,5)))
            }

class ImageAugmentation(object):
    """
    A class for performing image augmentation through various transformations like rotation, zoom, etc.
    Allows randomization of parameters for each transformation and maintains a history of applied transformations.
    """

    
    def __init__(self, img: Union[str, np.ndarray] = None, 
                 min_max_parameters: Optional[dict] = None, 
                 multitr_chain: Optional[List[str]] = None) -> None:
        """
        Initializes the ImageAugmentation class with an image and optional transformation parameters.

        Parameters:
        ----------
        img : str or ndarray
            The path to an image file or the image data itself.
        min_max_parameters : dict, optional
            Custom min and max values for randonmly choose parameter transformation.
        multitr_chain : list, optional
            A predefined chain of transformations to apply.
        """
        
        self._transformparameters = {}
        self._new_images = {}
        self.tr_paramaters = {}
        self._min_max_parameters = None
        self.img_data = None
        if img is not None:
            self.img_data = cv2.imread(img) if isinstance(img, str) else copy.deepcopy(img)
        if min_max_parameters:
            self._min_max_parameters = {k: np.linspace(v[0],v[1],20) for k,v in min_max_parameters.items()}

        self._multitr_chain = multitr_chain
        
    
    @property
    def available_transforms(self):
        return list(self._run_default_transforms.keys())
    
    @property
    def _run_default_transforms(self):
        
        return  {
                'rotation': self.rotate_image,
                'denoise': self.denoise_image,
                'zoom': self.expand_image,
                'clahe': self.clahe,
                'shift': self.shift_ndimage,
                'multitr': self.multi_transform,
                'flip': self.flip_image,
                'hsv': self.hsv,
                'shear': self.shear_image,
                'perspective': self.perspective_image,
                'gaussian': self.diff_gaussian_image,
                'illumination':self.change_illumination    
            }
        
    @property    
    def params_kwargs(self):
        return {
            'rotation': 'angle',
            'zoom': 'zoom_factor',
            'clahe': 'thr_constrast',
            'shear': ['shear_x','shear_y'],
            'shift': ['xshift','yshift'],
            'flip': 'flipcode',
            'gaussian': 'high_sigma',
            'hsv': 'hsvparams',
            'illumination': 'illuminationparams'
            }

    @property
    def _augmented_images(self):
        return self._new_images

    @property
    def _random_parameters(self):

        if self._min_max_parameters is None: return DEFAULT_PARAMS
        else: return {k: random.choice(v) for k,v in self._min_max_parameters.items()}        
        
    
    def _select_random_transforms(self, n_chains = 4):
        """
        Randomly selects a set of transformations for multi-transform.

        Returns:
        -------
        list
            A list of randomly selected transformation names.
        """
        chain_transform = []
        while len(chain_transform) < n_chains:
            trname = random.choice(list(self._run_default_transforms.keys()))
            if trname != 'multitr' and trname not in chain_transform:
                chain_transform.append(trname)
                
        return chain_transform

    def updated_paramaters(self, tr_type):
        """
        this function updates the tranformation dictionary information
        
        Parameters:
        ----------
        tr_type : str, optional
            transformation name
        """
        self.tr_paramaters.update({tr_type : self._transformparameters[tr_type]})
    
    
    #['flip','zoom','shift','rotation']
    def multi_transform(self, img: Union[str, np.ndarray] = None, 
                        chain_transform: Optional[List[str]] = None,
                         params: Optional[dict] = None, update: bool = True) -> np.ndarray:

        """
        Applies a chain of multiple transformations to the image.

        Parameters:
        ----------
        img : ndarray, optional
            The image to transform. Uses the class's internal image data if None.
        chain_transform : list, optional
            A list of transformation names to apply.
        params : dict, optional
            Parameters for each transformation in the chain.
        update : bool, optional
            If True, updates the class's internal state with the result.

        Returns:
        -------
        ndarray
            The transformed image.
        """
        
        # Selecting transformations if not provided
        if chain_transform is None:
            chain_transform = self._multitr_chain if self._multitr_chain is not None else self._select_random_transforms()
                 
        if img is None:
            img = self.img_data

        imgtr = copy.deepcopy(img)
        augmentedsuffix = {}
        
        for transform_name in chain_transform:
            if params is None:
                imgtr = perform_kwargs(self._run_default_transforms[transform_name],
                     img = imgtr,
                     update = False)
            else:
                
                imgtr = perform(self._run_default_transforms[transform_name],
                     imgtr,
                     params[transform_name], False)
            #if update:
            augmentedsuffix[transform_name] = self._transformparameters[transform_name]
        
        self._transformparameters['multitr'] = augmentedsuffix
         
        if update:
            
            self.updated_paramaters(tr_type = 'multitr')
            self._new_images['multitr'] = imgtr

        return imgtr


    def apply_transformation(self, transform_func, img=None, transform_param=None, 
                             transform_name=None, update=True):
        """
        Applies a specific image transformation function.

        Parameters:
        ----------
        transform_func : function
            The specific transformation function to apply.
        img : ndarray, optional
            The image to be transformed. If None, uses the class's internal image data.
        transform_param : various types, optional
            The parameter specific to the transformation.
        transform_name : str, optional
            The name of the transformation (for internal tracking).
        update : bool, optional
            If True, updates the class's internal state with the result.

        Returns:
        -------
        ndarray
            The transformed image.
        """

        if img is None:
            img = self.img_data

        if transform_param is None and transform_name:
            transform_param = self._random_parameters.get(transform_name, None)

        # Validate that the function is callable
        if not callable(transform_func):
            raise ValueError("Provided function is not callable.")
        img_transformed = transform_func(img, transform_param)

        if update and transform_name:
            self._transformparameters[transform_name] = transform_param
            self.updated_paramaters(tr_type=transform_name)
            self._new_images[transform_name] = img_transformed

        return img_transformed

    def diff_gaussian_image(self, img: Union[str, np.ndarray] = None, 
                            high_sigma: Optional[float] = None, update: bool = True):
        """
        Applies a differential Gaussian filter to the image.

        Parameters:
        ----------
        img : ndarray, optional
            The image to be processed. If None, uses the class's internal image data.
        high_sigma : float, optional
            The standard deviation for the Gaussian kernel. If None, a random value is chosen based on class parameters.
        update : bool, optional
            If True, updates the class's internal state with the result.

        Returns:
        -------
        ndarray
            The image after applying the differential Gaussian filter.
        """
        
        if img is None:
            img = copy.deepcopy(self.img_data)
            
        if high_sigma is None:
            high_sigma = self._random_parameters['gaussian']
        
        imgtr,_ = adding_noise(img,sigma = high_sigma)
        self._transformparameters['gaussian'] = high_sigma
        
        if update:
            
            self.updated_paramaters(tr_type = 'gaussian')
            self._new_images['gaussian'] = imgtr

        return imgtr
    
    def denoise_image(self, img: Union[str, np.ndarray] = None, 
                            sigma: Optional[float] = None, update: bool = True):
        """
        Applies a denoise_image

        Parameters:
        ----------
        img : ndarray, optional
            The image to be processed. If None, uses the class's internal image data.
        high_sigma : float, optional
            The standard deviation for the Gaussian kernel. If None, a random value is chosen based on class parameters.
        update : bool, optional
            If True, updates the class's internal state with the result.

        Returns:
        -------
        ndarray
            The image after applying the differential Gaussian filter.
        """
        
        if img is None: img = copy.deepcopy(self.img_data)
            
        if sigma is None: sigma = self._random_parameters['denoise']

        imgtr,_ = applying_denoise(img,sigma = sigma)
        self._transformparameters['denoise'] = sigma
        
        if update:
            
            self.updated_paramaters(tr_type = 'denoise')
            self._new_images['denoise'] = imgtr

        return imgtr
    
    
    def perspective_image(self, img: np.ndarray = None, perspective_x: float = None, perspective_y: float = None,update:bool = True):
        """
        Apply a perspective transformation to an image.

        Parameters:
        ----------
        img : np.ndarray
        The input image to be transformed.
        perspective_x : float, optional
            The perspective transformation factor along the x-axis.
        perspective_y : float, optional
            The perspective transformation factor along the y-axis.
        update : bool, optional
            If True, updates the class's internal state with the result.

        Returns:
        -------
        ndarray:
            The rotated image.

        Raises:
        ------
        ValueError:
            If the input image is not in the expected format or dimensions.
        """
        
        if img is None:
            img = copy.deepcopy(self.img_data)
        
        if perspective_x is None:
            perspective_x, _ = self._random_parameters['perspective']
        if perspective_y is None:
            _, perspective_y = self._random_parameters['perspective']

        
        imgtr = perspective_transform(img,perspective_x = perspective_x, perspective_y=perspective_y)
        self._transformparameters['perspective'] = [perspective_x, perspective_y]
        
        if update:
            
            self.updated_paramaters(tr_type = 'perspective')
            self._new_images['perspective'] = imgtr

        return imgtr
    
    def shear_image(self, img: np.ndarray = None, shear_x: float = None, shear_y:float = None,update:bool = True):
        """
        Shear the given image by a specified angle.

        Parameters:
        ----------
        img : ndarray, optional
            The image to be rotated. If None, uses the class's internal image data.
        shear_x : float, optional
            The shear factor for shear the image in the x axis. the values must be between 0 to 1
        shear_y : float, optional
            The shear factor for shear the image in the y axis. the values must be between 0 to 1
        update : bool, optional
            If True, updates the class's internal state with the result.

        Returns:
        -------
        ndarray:
            The rotated image.

        Raises:
        ------
        ValueError:
            If the input image is not in the expected format or dimensions.
        """
        
        if img is None:
            img = copy.deepcopy(self.img_data)
        
        if shear_x is None:
            shear_x, _ = self._random_parameters['shear']
        if shear_y is None:
            _, shear_y = self._random_parameters['shear']

        
        imgtr = shear_image(img,shear_x = shear_x, shear_y=shear_y)
        self._transformparameters['shear'] = [shear_x, shear_y]
        
        if update:
            
            self.updated_paramaters(tr_type = 'shear')
            self._new_images['shear'] = imgtr

        return imgtr
    
    def rotate_image(self, img = None, angle = None, update = True):
        """
        Rotates the given image by a specified angle.

        Parameters:
        ----------
        img : ndarray, optional
            The image to be rotated. If None, uses the class's internal image data.
        angle : int, optional
            The angle in degrees for rotating the image. If None, a random angle is chosen based on class parameters.
        update : bool, optional
            If True, updates the class's internal state with the result.

        Returns:
        -------
        ndarray:
            The rotated image.

        Raises:
        ------
        ValueError:
            If the input image is not in the expected format or dimensions.
        """
        
        if img is None:
            img = copy.deepcopy(self.img_data)
        if angle is None:
            angle = self._random_parameters['rotation']

        
        imgtr = image_rotation(img,angle = angle)
        self._transformparameters['rotation'] = angle
        
        if update:
            
            self.updated_paramaters(tr_type = 'rotation')
            self._new_images['rotation'] = imgtr

        return imgtr

    def flip_image(self, img = None, flipcode = None, update = True):

        if img is None:
            img = copy.deepcopy(self.img_data)
        if flipcode is None:
            flipcode = self._random_parameters['flip']


        imgtr = image_flip(img,flipcode = int(flipcode))
        
        self._transformparameters['flip'] = flipcode
        
        if update:
            
            self.updated_paramaters(tr_type = 'flip')
            self._new_images['flip'] = imgtr

        return imgtr

    def expand_image(self, img = None, ratio = None, update = True):
        if ratio is None:
            ratio = self._random_parameters['zoom']
            
        if img is None:
            img = copy.deepcopy(self.img_data)
            
        imgtr = image_zoom(img, zoom_factor=ratio)
        
        self._transformparameters['zoom'] = ratio
        if update:
            
            self.updated_paramaters(tr_type = 'zoom')
            self._new_images['zoom'] = imgtr

        return imgtr
    

    def shift_ndimage(self,img = None, shift  = None, update = True):

        max_displacement = None
        if shift is None:
            max_displacement = (self._random_parameters['shift'])/100
        if img is None:
            img = copy.deepcopy(self.img_data)
        
        imgtr, displacement =  randomly_displace(img, 
                                                 maxshift = max_displacement, 
                                                 xshift = shift, yshift = shift)
        
        self._transformparameters['shift'] = displacement[0]
        if update:
            
            self.updated_paramaters(tr_type = 'shift')
            self._new_images['shift'] = imgtr#.astype(np.uint8)

        return imgtr#.astype(np.uint8)
    
    def change_illumination(self, img = None, illuminationparams =None, update = True):
        if img is None:
            img = copy.deepcopy(self.img_data)
        if illuminationparams is None:
            illuminationparams = self._random_parameters['illumination']
        
        
        cv2img = cv2_array_type(img)
           
        imgtr = illumination_shift(cv2img,valuel = illuminationparams)
        
        if img.shape[0] == 3:
            imgtr = imgtr.swapaxes(2,1).swapaxes(1,0)
        if np.nanmax(img) < 2:
            imgtr = imgtr/255.
            
        self._transformparameters['illumination'] = illuminationparams
        if update:
            
            self.updated_paramaters(tr_type = 'illumination')
            self._new_images['illumination'] = imgtr

        return imgtr
    
    def clahe(self, img= None, thr_constrast = None, update = True):

        if thr_constrast is None:
            thr_constrast = self._random_parameters['clahe']/10
        
        if img is None:
            img = copy.deepcopy(self.img_data)
        
        cv2img = cv2_array_type(img)
        
        imgtr,_ = clahe_img(cv2img, clip_limit=thr_constrast)
        
        if img.shape[0] == 3:
            imgtr = imgtr.swapaxes(2,1).swapaxes(1,0)
        if np.nanmax(img) < 2:
            imgtr = imgtr/255.
            
        self._transformparameters['clahe'] = thr_constrast
        if update:
            
            self.updated_paramaters(tr_type = 'clahe')
            self._new_images['clahe'] = imgtr
            
        return imgtr

    def hsv(self, img = None, hsvparams =None, update = True):
        if img is None:
            img = copy.deepcopy(self.img_data)
        if hsvparams is None:
            hsvparams = self._random_parameters['hsv']
        
        cv2img = cv2_array_type(img)
        
        imgtr,_ = shift_hsv(cv2img,hue_shift=hsvparams[0], sat_shift = hsvparams[1], val_shift = 0)
        
        if img.shape[0] == 3:
            imgtr = imgtr.swapaxes(2,1).swapaxes(1,0)
        if np.nanmax(img) < 2:
            imgtr = imgtr/255.
        
        self._transformparameters['hsv'] = hsvparams
        if update:
            
            self.updated_paramaters(tr_type = 'hsv')
            self._new_images['hsv'] = imgtr

        return imgtr
    
    def random_augmented_image(self,img= None, update = True):
        if img is None:
            img = copy.deepcopy(self.img_data)
        
        imgtr = copy.deepcopy(img)
        augfun = random.choice(list(self._run_default_transforms.keys()))
        
        imgtr = perform_kwargs(self._run_default_transforms[augfun],
                     img = imgtr,
                     update = update)

        return imgtr

    def _transform_as_ids(self, tr_type):

        if type (self.tr_paramaters[tr_type]) ==  dict:
            paramsnames= ''
            for j in list(self.tr_paramaters[tr_type].keys()):
                paramsnames += 'ty_{}_{}'.format(
                    j,
                    summarise_trasstring(self.tr_paramaters[tr_type][j]) 
                )

        else:
            paramsnames = summarise_trasstring(self.tr_paramaters[tr_type])

        return '{}_{}'.format(
                tr_type,
                paramsnames
            )
    

class MultiTimeTransformer():
    """A transformer for applying multiple spatial and noise-based transformations to multi-dimensional images.

    Handles both input images and segmentation masks with proper dimension management and transformation
    parameter consistency.

    Attributes
    ----------
    data_format : str
        The data format convention for input tensors (default: 'CDHW' - Channel, Depth, Height, Width)
    _available : List[str]
        List of available transformation types

    Parameters
    ----------
    transformer : object
        Transformer object containing specific transformation methods
    data_format : str, optional
        Data format convention, by default 'CDHW'
    available_transforms : Optional[List[str]], optional
        List of allowed transformations, by default None (uses all available)
    """

    dataformat_orders: Dict[str, List[str]] = {
        'perspective': ['CHW->HWC', 'HWC->CHW'],
        'shear': ['CHW->HWC', 'HWC->CHW'],
        'rotation': ['CHW->HWC', 'HWC->CHW'],
        'flip': ['CHW->HWC', 'HWC->CHW'],
        'zoom': ['CHW->HWC', 'HWC->CHW'],
        'shift': ['CHW->HWC', 'HWC->CHW'],
        'gaussian': ['CHW->CHW', 'CHW->CHW'],
        'denoise': ['CHW->CHW', 'CHW->CHW'],
    }
    
        
    def __init__(
        self,
        transformer: ImageAugmentation,
        data_format: str = 'CDHW',
        available_transforms: Optional[List[str]] = None
    ) -> None:
        self.data_format = data_format
        self._transformer = copy.deepcopy(transformer)
        self._available = available_transforms or list(self.dataformat_orders.keys())
        self.reset_transformer()
    
    @staticmethod
    def check_dimensions( image):
        """Ensure input has 4 dimensions (CDHW format).

        Parameters
        ----------
        image : np.ndarray
            Input image tensor

        Returns
        -------
        np.ndarray
            Output image tensor with guaranteed 4 dimensions
        """
        if len(image.shape) == 3:
            image = image.expand_dims(axis = 0)
            
        return image  
    
    def tr_functions(self):
        return {
        'perspective': self.transformer.perspective_image,
        'shear': self.transformer.shear_image,
        'rotation': self.transformer.rotate_image,
        'flip': self.transformer.flip_image,
        'shift': self.transformer.shift_ndimage,
        'zoom': self.transformer.expand_image,
        'gaussian': self.transformer.diff_gaussian_image,
        'denoise': self.transformer.denoise_image
        }
    
    def reset_transformer(self):
        self.transformer = copy.deepcopy(self._transformer)
    


    
    def perspective(self,image, fun_name = 'perspective', **kwargs):
        return self.transform_image(image, fun_name=fun_name, **kwargs)
    
    def shear(self,image, fun_name = 'shear', **kwargs):
        return self.transform_image(image, fun_name=fun_name, **kwargs)
    
    def rotation(self,image, fun_name = 'rotation', **kwargs):
        return self.transform_image(image, fun_name=fun_name, **kwargs)
    
    def flip(self,image, fun_name = 'flip', **kwargs):
        return self.transform_image(image, fun_name=fun_name, **kwargs)
    
    def zoomin(self,image, fun_name = 'zoom', **kwargs):
        return self.transform_image(image, fun_name=fun_name, **kwargs)
    
    def guassian_noise(self,image, fun_name = 'gaussian', **kwargs):
        return self.transform_image(image, fun_name=fun_name, **kwargs)
    
    def denoise_image(self,image, fun_name = 'denoise', **kwargs):
        return self.transform_image(image, fun_name=fun_name, **kwargs)
    
    
    def transform_image(
        self,
        image: np.ndarray,
        fun_name: str,
        **kwargs
    ) -> np.ndarray:
        """Apply specified transformation to input image.

        Parameters
        ----------
        image : np.ndarray
            Input image in CDHW format
        fun_name : str
            Name of transformation to apply
        **kwargs
            Additional transformation-specific parameters

        Returns
        -------
        np.ndarray
            Transformed image with same dimensions as input

        Raises
        ------
        ValueError
            If unknown transformation name is provided
        """
        if fun_name not in self.tr_functions():
            raise ValueError(f"Unknown transformation: {fun_name}. Valid options: {list(self.tr_functions().keys())}")
        
        datatransformer = np.zeros_like(image)
        datatransformer[:] = image[:]
        datatransformer = self.check_dimensions(datatransformer)
        self.reset_transformer()
                
        for i in range(datatransformer.shape[0]):
            if i == 0:
                trimg = self.tr_functions()[fun_name](
                    np.einsum(self.dataformat_orders[fun_name][0], datatransformer[i]), **kwargs)
            else:
                params = self.transformer._transformparameters[fun_name] if isinstance(self.transformer._transformparameters[fun_name],list) else [self.transformer._transformparameters[fun_name]]
                trimg = self.tr_functions()[fun_name](
                    np.einsum(self.dataformat_orders[fun_name][0], datatransformer[i]), 
                    *params
                    )
            datatransformer[i] = np.einsum(self.dataformat_orders[fun_name][1], trimg)
        
        return datatransformer
    
    def __call__(
        self,
        image: np.ndarray,
        nmax_transforms: Optional[int] = None
    ) -> np.ndarray:
        """Apply random sequence of transformations to input image.

        Parameters
        ----------
        image : np.ndarray
            Input image tensor
        nmax_transforms : Optional[int], optional
            Maximum number of transformations to apply, by default 1

        Returns
        -------
        np.ndarray
            Transformed image tensor
        """
        
        self._params = {}
        transformed_img = np.zeros(image.shape, dtype=image.dtype)
        transformed_img[:] = image[:]
        nmax_transforms = nmax_transforms or 1
        options = set(random.choice(self._available + ['raw']) for _ in range(nmax_transforms))
        
        for transform in options:
            if transform != 'raw':
                transformed_img = self.transform_image(transformed_img, fun_name=transform)
                self._params[transform] = self.transformer._transformparameters[transform]
            else:
                self._params[transform] = 0
                break
        return transformed_img
    
class MTSegmenTransformer(MultiTimeTransformer):
    
    def __init__(self, transformer, data_format='CDHW', available_transforms=None):
        super().__init__(transformer, data_format, available_transforms)
        
    def transform_inputs(self,
        input_image: np.ndarray,
        target_image: np.ndarray,
        nmax_transforms: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply coordinated transformations to input-target pair.

        Parameters
        ----------
        input_image : np.ndarray
            Input image
        target_image : np.ndarray
            Segmentation mask image 2D
        nmax_transforms : int, optional
            Maximum number of transformations, by default 1

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Transformed input and target tensors

        Notes
        -----
        - Uses nearest-neighbor interpolation for segmentation masks
        - Maintains transformation parameter consistency between input and target
        """
        #assert len(self._params)>0, "Apply tranformation to the input first"
        tr_input_image = self(input_image, nmax_transforms)
        tr_target = np.copy(target_image)
        
        self.reset_transformer()
                
        for transform, params in self._params.items():
            
            if transform in ['denoise', 'gaussian', 'raw']:
                continue
            if params is not None:
                params = params if isinstance(params,list) else [params]
                tr_target = self.tr_functions()[transform](
                    tr_target,#np.einsum(mlt_transformer.dataformat_orders[k][0], datatransformer), 
                    *params
                    )

        return tr_input_image, tr_target
                


