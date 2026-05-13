import torch

from .engine import DLBaseEngine, sobel_filter


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