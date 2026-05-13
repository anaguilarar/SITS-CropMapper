
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import random
import skimage.transform as trans
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

from . import HLSSCALERPARMS, HLSSCALERPARMS_N, HLSSCALERPARMS_7C


import warnings
warnings.simplefilter(action="ignore")

def transform_keypoints(kps, meta, invert=False):
    keypoints = kps.copy()
    if invert:
        meta = np.linalg.inv(meta)
    keypoints[:, :2] = np.dot(keypoints[:, :2], meta[:2, :2].T) + meta[:2, 2]
    return keypoints

def normalize_transform(transform_matrix):
    #https://github.com/wuneng/WarpAffine2GridSample/blob/master/main.py
    src = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.float32)
    dst = transform_keypoints(src, transform_matrix)

    src = src / [48, 48] * 2 - 1
    dst = dst / [48, 48] * 2 - 1
    return trans.estimate_transform("affine", src=dst, dst=src).params

def perform_affine_tf(data, tf_matrices):

    # expects 4D tensor, we preserve gradients if there are any

    n_i, k, h, w = data.shape
    n_i2, r, c = tf_matrices.shape
    assert (n_i == n_i2)
    assert (r == 2 and c == 3)

    grid = F.affine_grid(tf_matrices, data.shape)  # output should be same size
    data_tf = F.grid_sample(data, grid,
                            padding_mode="zeros", align_corners=False)  # this can ONLY do bilinear

    return data_tf

def get_affine_from_random(min_rot=None, max_rot=None, min_shear=None,
                max_shear=None, min_scale=None, max_scale=None):
    
    a = np.radians(np.random.rand() * (max_rot - min_rot) + min_rot)
    shear = np.radians(np.random.rand() * (max_shear - min_shear) + min_shear)
    scale = np.random.rand() * (max_scale - min_scale) + min_scale

    return np.array([[np.cos(a) * scale, - np.sin(a + shear) * scale, 0.],
                            [np.sin(a) * scale, np.cos(a + shear) * scale, 0.],
                            [0., 0., 1.]], dtype=np.float32)  # 3x3

def get_transform(center, scale,  output_size,shear = 0, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(output_size[1]) / h
    t[1, 1] = float(output_size[0]) / h
    t[0, 2] = output_size[1] * (-float(center[0]) / h + .5)
    t[1, 2] = output_size[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad+shear), np.cos(rot_rad+shear)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -output_size[1] / 2
        t_mat[1, 2] = -output_size[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        
    return t

def random_affine_mod(img, min_rot=None, max_rot=None, min_shear=None,
                    max_shear=None, min_scale=None, max_scale=None):
    """
    modified from https://github.com/xu-ji/IIC/blob/master/code/utils/segmentation/transforms.py
    """
    
    rot = random.randint(min_rot, max_rot)
    height, width = img.shape[-2:]
    center = [height / 2., width / 2.]
    scale = max(height, width) / 200. * random.uniform(min_scale, max_scale)
    shear = np.radians(np.random.rand() * (max_shear - min_shear) + min_shear)
    output_size = (width, height)
    affine1 = get_transform(center, scale, output_size,shear, rot)
    affine1 = normalize_transform(affine1).astype(np.float32)
    inv_affine1 = np.linalg.inv(affine1).astype(np.float32)
    
    affine1, inv_affine1 = affine1[:2, :],inv_affine1[:2, :]
    affine1, inv_affine1 = torch.from_numpy(affine1), torch.from_numpy(inv_affine1)
    
    img = perform_affine_tf(img.unsqueeze(dim=0), affine1.unsqueeze(dim=0))
    img = img.squeeze(dim=0)

    return img, affine1, inv_affine1


def sobel_filter(image, device = None):
    device = device or "cuda:0" if torch.cuda.is_available() else "cpu"
    #modified from https://github.com/xu-ji/IIC/blob/master/code/utils/cluster/transforms.py#L47
    bn,t,c, h, w = image.size()
    
    gray_img = torch.movedim(image[:,:,-2].unsqueeze(2),1,2)
    
    sobel1 = np.array([[[1,0,-1],[2,0,-2],[1,0,-1]]])
    conv1 = nn.Conv3d(1,1,kernel_size = (1,1,1), stride = (1,1,1), padding = (0,1,1), bias  =False)
    conv1.weight = nn.Parameter(
    torch.Tensor(sobel1).float().unsqueeze(0).unsqueeze(0).to(device))
    dx = conv1(Variable(gray_img)).data
    
    sobel2 = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]])
    
    conv2 = nn.Conv3d(1,1,kernel_size = (1,1,1), stride = (1,1,1), padding = (0,1,1), bias  =False)
    conv2.weight = nn.Parameter(
        torch.Tensor(sobel2).float().unsqueeze(0).unsqueeze(0).to(device))
    dy = conv2(Variable(gray_img)).data

    sobel_imgs = torch.cat([dx, dy], dim=1)
    sobel_imgs = torch.movedim(sobel_imgs, 1,2)

    return torch.cat([image[:,:,:-1], sobel_imgs, image[:,:,-1].unsqueeze(2)], dim=2)

class AffineTransform():
    
    min_rot = -90
    max_rot = 90
    min_shear = -10
    max_shear = 10
    min_scale = 0.8
    max_scale = 1.2

    def __init__(self):
        
        self.affine_kwargs = {"min_rot": self.min_rot, 
                        "max_rot": self.max_rot,
                        "min_shear": self.min_shear,
                        "max_shear": self.max_shear,
                        "min_scale": self.min_scale,                     
                        "max_scale": self.max_scale}
    
    def __call__(self, img):
        """
        img dimensions D x C x W x H
        """
        assert len(img.shape) == 4
        n_tps = img.shape[0]
        imgnpp = np.zeros_like(img)
        imgnpp = torch.from_numpy(imgnpp).to(torch.float32)
        
        imgnpp[0], affine1_to_2, affine2_to_1 = random_affine_mod(img[0], **self.affine_kwargs)
        
        for i in range(1, n_tps):
            imgnpp[i] = perform_affine_tf(img[i].unsqueeze(dim=0), affine1_to_2.unsqueeze(dim=0))

        return imgnpp, affine2_to_1



class Normalize(object):
    """
    """
    def __init__(self, dataset, nbands = 5):
        # ['blue', 'green', 'red', 'nir', 'ndvi', 'gndvi']
        if dataset == 'hls':
            if nbands == 5:
                self.mean_fold1 = np.array([HLSSCALERPARMS_N['mean']+[[[0]]]]).astype(np.float32)
                self.std_fold1 = np.array([HLSSCALERPARMS_N['std']+[[[1]]]]).astype(np.float32)
            elif nbands == 7:
                self.mean_fold1 = np.array([HLSSCALERPARMS_7C['mean']+[[[0]]]]).astype(np.float32)
                self.std_fold1 = np.array([HLSSCALERPARMS_7C['std']+[[[1]]]]).astype(np.float32)

        # TODO others
            
    def __call__(self, sample):
        # print('mean: ', sample['img'].mean(dim=(0,2,3)))
        # print('std : ', sample['img'].std(dim=(0,2,3)))
        datazeros = sample == 0
        sample = (sample - self.mean_fold1) / self.std_fold1
        sample[datazeros] = 0
        return sample


class ToTensor(object):
    def __init__(self):
        pass
    
    def __call__(self, sat_data):
        
        sat_data_tensor = torch.from_numpy(sat_data).to(torch.float32)#.unsqueeze(0)
        
        return sat_data_tensor
    
    
class TemporalCutMix:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        """
        Args:
            x: (B, T, C, H, W) - input time-series images
            y: (B, T, H, W) - segmentation masks across time
        Returns:
            mixed_x: (B, T, C, H, W)
            mixed_y: (B, T, H, W)
        """
        batch_size = x.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        indices = torch.randperm(batch_size)
        
        # Same spatial region across all time steps
        h, w = x.shape[-2:]
        cx, cy = np.random.uniform(0, w), np.random.uniform(0, h)
        ww, hh = int(w * np.sqrt(1 - lam)), int(h * np.sqrt(1 - lam))
        x1 = int(np.clip(cx - ww//2, 0, w))
        y1 = int(np.clip(cy - hh//2, 0, h))
        x2 = int(np.clip(cx + ww//2, 0, w))
        y2 = int(np.clip(cy + hh//2, 0, h))

        # Apply same cut to all time steps
        mixed_x = x.clone()
        mixed_x[:, :, :, y1:y2, x1:x2] = x[indices, :, :, y1:y2, x1:x2]
        
        mixed_y = y.clone()
        mixed_y[:, :, y1:y2, x1:x2] = y[indices, :, y1:y2, x1:x2]

        return mixed_x, mixed_y
    
    
class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, kernel_size =9, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)

        
class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        n_bands = 7,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                #transforms.RandomApply(
                #    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                #    p=0.8,
                #),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = GaussianBlur(p=0.1)

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = Normalize('hls', n_bands)

        self.global_transfo1 = global_transfo1_extra
        
        self.global_transfo2 = global_transfo2_extra
        self.local_transfo = local_transfo_extra

    def __call__(self, sat_image):
        
        output = {}
        dates = sat_image[:,-1]
        #sat_image = transforms.ToTensor()(sat_image[:,:-1])
        satdata = sat_image[:,:-1]
        sat_image = ToTensor()(satdata)

        #satdata_g = ToTensor()(satdata)
        dates = torch.from_numpy(dates).unsqueeze(dim=1).to(torch.float32)

        # global crops:
        im1_base = self.geometric_augmentation_global(sat_image)
        global_crop_1 = self.global_transfo1(im1_base)
        global_crop_1 = torch.concat([global_crop_1, dates], dim = 1)
        global_crop_1 = self.normalize(global_crop_1)

        im2_base = self.geometric_augmentation_global(sat_image)
        global_crop_2 = self.global_transfo2(im2_base)
        global_crop_2 = torch.concat([global_crop_2, dates], dim = 1)
        global_crop_2 = self.normalize(global_crop_2)
        

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = []
        for _ in range(self.local_crops_number):
            local_crop = self.local_transfo(self.geometric_augmentation_local(sat_image)) 
            local_crop = torch.concat([local_crop, dates[:,:,:self.local_crops_size,:self.local_crops_size ]], dim = 1)
            local_crop = self.normalize(local_crop)
            local_crops.append(local_crop)

        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output

      
class AddRandomGaussianNoise(object):
    def __init__(self, mean=0., std_range=(0.01, 0.05)): # std_range relative to 0-1 normalized data
        self.mean = mean#torch.from_numpy(np.array(mean))
        self.std_range = std_range

    def __call__(self, tensor): # expects (T, C, H, W)
        std = torch.rand(1).item() * (self.std_range[1] - self.std_range[0]) + self.std_range[0]
        
        
        time_mask = (~(tensor==0)).float()[:, 0, :, :].unsqueeze(dim=1)
        zero_channels = torch.all(time_mask == 0, dim=( 2,3)).squeeze()
        notvalid_time_indices  = torch.where(zero_channels)[0]
        
        ## filter those dates that does not have data
        retain_time_indices = [i for i in range(tensor.shape[0]) if i not in notvalid_time_indices ]
        output_tensor = tensor.clone()
        noise = torch.randn(output_tensor[retain_time_indices].size()) * std + self.mean
        output_tensor[retain_time_indices] = output_tensor[retain_time_indices] + noise.to(tensor.device, tensor.dtype)
        return output_tensor
    

class DataAugmentationDINOV2(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        rgb_indexes = [2,1,0],
        #spectral_scale = (0.85,1.15),
        std_range = (0.001,0.005),
        n_bands = 7,
    ):  
        
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.rgb_indexes = rgb_indexes
        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p = 0.5),
                transforms.RandomRotation(degrees=(0, 180))
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p = 0.5),
                transforms.RandomRotation(degrees = (0,180))
            ]
        )

        self.color_jittering =  transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                )
        
        self.spectral_jittering = transforms.RandomApply(
            [AddRandomGaussianNoise(std_range=std_range)],
            p=0.5 # Example probability, tune as needed
        )
        #AddRandomGaussianNoise(std_range =std_range)
        global_transfo1_extra = GaussianBlur(p=1.0, radius_min=0.1,radius_max=2.0)

        global_transfo2_extra = GaussianBlur(p=0.1, radius_min=0.1, radius_max=2.0)

        local_transfo_extra = GaussianBlur(p=0.5, radius_min=0.1, kernel_size=5, radius_max=0.8) 

        # normalization
        self.normalize = Normalize('hls', n_bands)

        self.global_transfo1 = global_transfo1_extra
        
        self.global_transfo2 = global_transfo2_extra
        self.local_transform = local_transfo_extra
    
    def apply_transform_to_bands(self, transform, tensor_t_c_h_w, band_indices):
        if not band_indices:
            return tensor_t_c_h_w # Return unmodified if no bands selected

        selected_bands = tensor_t_c_h_w[:, band_indices]

        augmented_bands = transform(selected_bands)
        
        output_tensor = tensor_t_c_h_w.clone()
        output_tensor[:, band_indices] = augmented_bands
        return output_tensor
    
    def __call__(self, sat_image):
        
        output = {}
        dates = sat_image[:,-1,0,0]
        #sat_image = transforms.ToTensor()(sat_image[:,:-1])
        satdata = sat_image[:,:-1]
        sat_image = ToTensor()(satdata)
        
        all_indices = list(range(sat_image.shape[1])) # Channel dim index 1
        nonrgb_indices = [i for i in all_indices if i not in self.rgb_indexes]
        
        #rgb_images = sat_image[:, self.rgb_indexes]
        #nonrgb_images = sat_image[:, len(self.rgb_indexes):]

        #satdata_g = ToTensor()(satdata)
        date_channel_t = torch.from_numpy(dates).to(torch.float32).to(sat_image.device)
        expanded_date_gc = date_channel_t.unsqueeze(-1).unsqueeze(-1).expand(-1, self.global_crops_size, self.global_crops_size).unsqueeze(1)
        expanded_date_lc = date_channel_t.unsqueeze(-1).unsqueeze(-1).expand(-1, self.local_crops_size, self.local_crops_size).unsqueeze(1)
        # global crops:
        #photom_gc1_step1 = self.apply_transform_to_bands(self.color_jittering, sat_image, self.rgb_indexes)
        
        im1_base = self.geometric_augmentation_global(sat_image)
        #im1_base = self.apply_transform_to_bands(self.spectral_jittering, im1_base, all_indices)
        
        global_crop_1 = self.global_transfo1(im1_base)
        global_crop_1 = torch.concat([global_crop_1, expanded_date_gc], dim = 1)
        global_crop_1 = self.normalize(global_crop_1)
        

        #photom_gc2_step1 = self.apply_transform_to_bands(self.color_jittering, sat_image, self.rgb_indexes)
        im2_base = self.geometric_augmentation_global(sat_image)
        #im2_base = self.apply_transform_to_bands(self.spectral_jittering, im2_base, all_indices)
        global_crop_2 = self.global_transfo2(im2_base)
        global_crop_2 = torch.concat([global_crop_2, expanded_date_gc], dim = 1)
        global_crop_2 = self.normalize(global_crop_2)
        

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = []
        for _ in range(self.local_crops_number):
            local_crop = self.geometric_augmentation_local(sat_image)
            #photom_lc_step1 = self.apply_transform_to_bands(self.color_jittering, local_crop, self.rgb_indexes)
            #photom_lc_step1 = self.apply_transform_to_bands(self.spectral_jittering, local_crop, all_indices)
            local_crop = self.local_transform(local_crop)
            
            local_crop = torch.concat([local_crop, expanded_date_lc], dim = 1)
            local_crop = self.normalize(local_crop)
            local_crops.append(local_crop)

        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output