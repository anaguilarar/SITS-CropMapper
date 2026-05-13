import os

import torch

from models.engine import DLBaseEngine, sobel_filter
from models.dino_enginev2 import DINOTrainerSegmentationModel


class IICInferenceModel(DLBaseEngine):
    """
    Inference engine for deep learning models, designed to make predictions using a trained model.

    Parameters:
    ----------
    model : nn.Module
        The pre-trained neural network model.
    model_weight_path : str
        Path to the saved model weights.
    device : str, optional
        Device to which the model and data are sent ('cuda' or 'cpu'). Default is 'cuda' if available.
    sobel_filter : bool, optional
        Whether to apply a Sobel filter to input images before inference.
    """
    
    def __init__(self, model, optimizer, weight_path_dict, device=None, sobel_filter=True):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sobel_filter = sobel_filter
        super().__init__(model)
        # Load the model
        self.optimizer = optimizer
        if weight_path_dict is not None:
            self.load_weights(weight_path_dict)
        self.model.eval()
        
        
    def preprocess_input(self, image):
        """Preprocess a single image for inference."""
        image = image.to(self.device)
        if self.sobel_filter:
            image = sobel_filter(image)
        return image
    
    def predict(self, image):
        """
        Run inference on two images.
        
        Parameters:
        ----------
        image_1 : torch.Tensor
            First input image.
        image_2 : torch.Tensor
            Second input image.

        Returns:
        -------
        torch.Tensor
            Model predictions for the two images.
        """
        image_1 = self.preprocess_input(image)
        
        with torch.no_grad():
            output1 = self.model(image_1)
    
        return output1
    
class STViTS_detector(DINOTrainerSegmentationModel):
    
    def __init__(self, config, **kwargs):
        super().__init__(config, use_summary = config.TRAIN.get('summary_layer', False),**kwargs)
        
    
    def load_weights_for_detection(self, filepath):
        if not os.path.exists(filepath):
            print(f"Checkpoint file not found: {filepath}")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        
        print(f"Checkpoint loaded from {filepath}. ")
    
    
    def predict(self, x, x2 = None):
        self.model.eval()
        if self._summary_layer:
            x, x2 = x.to(self.device), x2.to(self.device)
        else:
            x= x.to(self.device)    
        
        with torch.no_grad():
            patch_tokens = self.intermediate_spatial_features(x)
        
            backbone_feats = self.model.backbone.norm_final(patch_tokens)
            if self._summary_layer:
                if self.config.MODEL.head_type == 'summconv2':
                    preds = self.model.head(backbone_feats, x2,
                        self.model.backbone.image_size, self.model.backbone.image_size)
                else:
                    preds = self.model.head(backbone_feats, x2,
                        self.model.backbone.image_size, self.model.backbone.image_size, self.config.MODEL.patch_size)
            else:
                preds = self.model.head(backbone_feats, 
                        self.model.backbone.image_size, self.model.backbone.image_size)
                
            
        return preds