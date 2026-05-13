import torch
import os
from datasets.transforms.tensor_transforms import sobel_filter
from utils.reporters import ReporterBase
from tqdm import tqdm
import logging
from .dino_utils import CosineScheduler,apply_optim_scheduler, update_teacher

from collections import OrderedDict
import numpy as np
import warnings
import torch.nn as nn
from models.TSViT import architectures
from functools import partial
from models.TSViT.module import DINOHead
from models.loss_functions import DINOLossV2 
import torch.nn.functional as F

from models.metrics.numpy_metrics import get_classification_metrics
def get_mean_metrics(logits, labels, n_classes, loss,  name=""):
    """
    :param logits: (N, D, H, W)
    """
    _, predicted = torch.max(logits.data, 1)
    # unique_predictions = predicted.unique().cpu().numpy()
    predicted = predicted.reshape(-1).cpu().numpy()
    
    labels = labels.reshape(-1).cpu().numpy()
    
    unk_masks = None
    acc, precision, recall, F1, IOU = get_classification_metrics(
        predicted, labels, n_classes, unk_masks)['micro']
    loss_ = float(loss.detach().cpu().numpy())
    return {"%sAccuracy" % name: float(acc), "%sPrecision" % name: float(precision), "%sRecall" % name: float(recall),
            "%sF1" % name: float(F1), "%sIOU" % name: float(IOU), "%sloss" % name: loss_}
    
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


class DLBaseEngine():
    """
    Base class for a deep learning training engine.

    This class defines the structure and required methods that every training engine must implement,
    including running iterations, saving models, and computing loss.

    Attributes
    ----------
    model : Torch.Model
        pyTorch base model
    iter : int
        Current iteration of the training loop.
    start_iter : int
        The iteration from which the training was started or resumed.

    Methods
    -------
    run_iter():
        A method to be implemented by subclasses to perform a single iteration of training.
    _save_loss(losses):
        Saves and logs losses from an iteration.
    save_model(path):
        Saves the model and optimizer states to a file.
    _write_metrics():
        Writes out current metric values using the configured reporter.
    compute_loss(pred, y):
        Computes the loss for a batch of predictions and ground truth labels.
    """
    
    def __init__(self, model) -> None:
        self.iter: int = 0
        self.start_iter: int = 0
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = model
        
        self.lr_scheduler= None
    
    def run_iter(self):
        """
        Performs a single iteration of the training or evaluation process.
        Must be overridden by subclasses.
        
        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _save_loss(self, losses):
        """
        Saves and logs losses from a training iteration.

        Parameters
        ----------
        losses : dict or torch.Tensor
            The losses to log, which can be a single torch.Tensor or a dictionary of loss components.
        """
        #detectron2
        if isinstance(losses, dict):
            metrics_dict = {k: v.detach().cpu().item() for k, v in losses.items()}
            for k,v in metrics_dict.items():
               exec('self.'+k + '=v')
        else:
            self.loss = losses.detach().cpu().item() 
    
    def save_model(self, path):
        """
        Saves the model and optimizer states to the specified path.

        Parameters
        ----------
        path : str
            Path to the directory where the model and optimizer states will be saved.

        Raises
        ------
        AssertionError
            If the specified path does not exist.
        """
        #assert os.path.exists(path), "The specified path does not exist."
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(),  path + '_scheduler_params')    
        torch.save(self.model.state_dict(),  path + '_model_params')
        torch.save(self.optimizer.state_dict(), path + '_optimizer_params')
        
        pathm = path + "_scaler_params"
        if self.grad_scaler:
            torch.save(self.grad_scaler.state_dict(), pathm)
    
    def load_weights(self, path_dict):
        """
        Loads the model, optimizer, and gradient scaler states from the specified paths.

        Parameters
        ----------
        path_dict : dict
            Directory that contains the paths each file from which the model state will be loaded.
        Raises
        ------
        FileNotFoundError
            If any of the specified files does not exist.
        """
        model_path = path_dict.get('model_path', None)
        model_state_dict = path_dict.get('model_state_dict', None)
        optimizer_path = path_dict.get('optimizer_path', None)
        scaler_path = path_dict.get('scaler_path', None)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model path {model_path} does not exist.")
        model_state_dict = torch.load(model_path, map_location=torch.device('cpu') )
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)

        if optimizer_path and os.path.exists(optimizer_path):
            optimizer_state_dict = torch.load(optimizer_path, map_location=torch.device(self.device))
            self.optimizer.load_state_dict(optimizer_state_dict)
        elif optimizer_path:
            raise FileNotFoundError(f"The optimizer path {optimizer_path} does not exist.")

        if scaler_path and os.path.exists(scaler_path):
            scaler_state_dict = torch.load(scaler_path)
            self.grad_scaler.load_state_dict(scaler_state_dict)
        elif scaler_path:
            raise FileNotFoundError(f"The scaler path {scaler_path} does not exist.")

        print("Model and other components (if specified) loaded successfully.")
        
        
    def _write_iter_metrics(self, evaluation = False):
        if evaluation:
            values = {k: self.__getattribute__(k) for k in self._iter_eval_reporter._report_keys}
            self._iter_eval_reporter.update_report(values)
        else:
            values = {k: self.__getattribute__(k) for k in self._iter_tr_reporter._report_keys}
            self._iter_tr_reporter.update_report(values)
            
    def _write_metrics(self):
        """
        Writes metrics using the reporter attribute if configured.
        """
        if self._reporter is None:
            return None
        
        values = {k: self.__getattribute__(k) for k in self._reporter._report_keys}
        self._reporter.update_report(values)
    
    def _init_lr_scheduler(self, scheduler = 'cosine'):
        if scheduler == 'lambda':
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(self._tr_data_loader) - 1)

            self.lr_scheduler = warmup_lr_scheduler(self.optimizer, 
                                            warmup_iters, warmup_factor)
        elif scheduler == 'cosine':
            from timm.scheduler.cosine_lr import CosineLRScheduler
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(self._tr_data_loader) - 1)
            
            t_initial = int(100 * len(self._tr_data_loader) / 1)
            warmup_steps = int(1 * len(self._tr_data_loader))

            self.lr_scheduler = CosineLRScheduler(
                self.optimizer,
                t_initial=t_initial,
                # t_mul=1.,
                lr_min=float(1e-6),
                warmup_lr_init=float(1e-8),
                warmup_t=warmup_steps,
                cycle_limit=int(2),
                t_in_epochs=False,
            )
    
    def _save_loss(self, losses, evaluation=False):
        """
        Save loss value from the iteration.

        Parameters:
        ----------
        loss : float
            Loss value from the current batch.
        evaluation : bool, optional
            Flag to determine if the loss is from evaluation.
        """
        
        #detectron2
        
        if isinstance(losses, dict):
            metrics_dict = {k: v.detach().cpu().item() for k, v in losses.items()}
            for k,v in metrics_dict.items():
                if evaluation:
                    exec('self.eval_'+k + '=v')
                else:
                    exec('self.'+k + '=v')
        else:
            if evaluation:
                self.eval_loss = losses.detach().cpu().item() 
            else:
                self.loss = losses.detach().cpu().item() 
            
    
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
    
    
class IICTrainerModel(DLBaseEngine):
    """
    Training engine for deep learning models incorporating various functionalities such as
    training, validation, logging, and gradient scaling.

    Parameters:
    ----------
    model : nn.Module
        The neural network model to train.
    train_data_loader : DataLoader
        DataLoader for training data.
    optimizer : torch.optim.Optimizer
        Optimizer used for training.
    validation_data_loader : DataLoader, optional
        DataLoader for validation data.
    reporter : Reporter, optional
        Tool to report metrics during training.
    grad_scaler : torch.cuda.amp.GradScaler, optional
        Gradient scaler for mixed precision training.
    loss_fcn : Callable, optional
        Loss function to be used during training.
    model_weight_path : str, optional
        Path to save the model weights.

    Attributes:
    ----------
    device : str
        Device to which the model and data are sent ('cuda' or 'cpu').
    """
    
    def __init__(self, model, train_data_loader,optimizer, validation_data_loader = None, 
                reporter = None,
                sobel_filter = True,
                grad_scaler= None, loss_fcn = None, 
                model_weight_path = None, weight_dict = None,
                reporter_losses = ['epoch', 'iter','loss']) -> None:
    
        super().__init__(model)
        self.sobel_filter = sobel_filter # IIC implementation
        self._tr_data_loader = train_data_loader
        self._val_data_loader = validation_data_loader
        self.optimizer = optimizer
        self._reporter = reporter
        self._weight_path = model_weight_path
        self._weight_dict = weight_dict
        self.grad_scaler = grad_scaler
        self._reporter_losses = reporter_losses
        self.loss_fcn = loss_fcn
        
        if grad_scaler is None:
            from torch.amp import GradScaler
            self.grad_scaler = GradScaler(
                init_scale=2.**16,  # Helps prevent underflow
                growth_interval=1000
            )
        else:
            self.grad_scaler = grad_scaler
            
        self.model = self.model.to(self.device)

        optimizer_to(self.optimizer,self.device)
        self.set_initial_params()
        self._set_initial_reporter_params()
    
    def set_initial_params(self):
        
        self.iter = 0
        self.eval_iter = 0
        self._data_loader_iter_obj = None
        self._data_loader_eval_iter_obj = None


    def _set_initial_reporter_params(self):
        self._iter_tr_reporter = ReporterBase()
        self._iter_tr_reporter.set_reporter(self._reporter_losses)
        self._iter_eval_reporter = ReporterBase()
        self._iter_eval_reporter.set_reporter(
            [self._reporter_losses[0]]+['eval_'+i for i in self._reporter_losses[1:]])

        if self._reporter is None:
            self._reporter = ReporterBase()
            self._reporter.set_reporter(self._reporter_losses)
            self._reporter.file_name = 'reporter.json'
        
    
    @property
    def _data_loader_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = iter(self._tr_data_loader)
        return self._data_loader_iter_obj
    
    @property
    def _data_loader_eval_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_eval_iter_obj is None and self._val_data_loader is not None:
            self._data_loader_eval_iter_obj = iter(self._val_data_loader)
        return self._data_loader_eval_iter_obj
        
    
    def fit(self, max_epochs: int, start_from: int = 0, 
            suffix_model: str = None,
            lag_best: int = None,
            start_saving_from =None):
        """
        Run the training process for a specified number of epochs.

        Parameters:
        ----------
        max_epochs : int
            Total number of epochs to train the model.
        start_from : int, optional
            The starting epoch number, useful for resuming training. Default is 0.
        """
        start_saving_from = start_saving_from or 0
        self.epoch = int(start_from) if start_from else 0
        suffix_model = suffix_model if suffix_model else ""
        lag_best = max_epochs if lag_best is None else lag_best
        logger = logging.getLogger(__name__)
        logger.info("Starting training from epoch {} to {}".format(self.epoch, max_epochs))
        
        pbar = tqdm(range(self.epoch, max_epochs),leave=True, desc="Overall Training Progress")
        #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 100)
        self._init_lr_scheduler()
        for _ in pbar:
            pbar.set_description("[Epoch %d]" % (self.epoch))

            
            self.train_one_epoch()
            if self.epoch % 2 == 0:
                self.lr_scheduler.step()
                #print("decaying lr, new lr: %.10f" % self.optimizer.param_groups[0]["lr"])
            
            epoch_metrics  = self.write_epoch_metrics(self.epoch)
            pbar.set_postfix(OrderedDict(epoch_metrics))
            self.epoch += 1
            self.set_initial_params() # Reset or update parameters if needed per epoch
            
            if self.epoch>start_saving_from:
                if self.epoch % 2 == 0:
                    outname = os.path.join(self._weight_path, self.model.model_name + suffix_model)
                    self.save_model(outname)
                #logging.info("The best model was saved at epoch: {} loss value: {:.4f}".format(self.epoch, bestloss))

        if self._weight_path is not None:            
            outname = os.path.join(self._weight_path, self.model.model_name+ '_last' + suffix_model)
            self.save_model(outname)

    def train_one_epoch(self):
        """
        Conduct training over one epoch.
        """
        self.model.train()
        
        max_iter = len(self._tr_data_loader)
        pbar = tqdm(range(max_iter), leave=True, colour='blue', desc="Training")
        for _ in pbar:
            #try:
                endprocess = self.run_iter()
                if endprocess:
                    toshow = {}
                    for k in self._iter_tr_reporter.report.keys():
                        toshow[k] = self._iter_tr_reporter.report[k][-1]
                    
                    pbar.set_postfix(OrderedDict(toshow))
            #except Exception:
            #    warnings.warn("Exception during training:")
                
    def run_iter(self):
        """
        Run a single training iteration.
        """
        
        assert self.model.training, "Model was changed to eval mode!"
        
        img_1, img_2, affine_xg_to_x = next(self._data_loader_iter)
        
        img_1 = img_1.to(self.device)
        img_2 = img_2.to(self.device)
        if self.sobel_filter:
            img_1 = sobel_filter(img_1)
            img_2 = sobel_filter(img_2)
            
        affine_xg_to_x = affine_xg_to_x.to(self.device)
        self.optimizer.zero_grad()
        with torch.amp.autocast(self.device):
            output1 = self.model(img_1)
            output2 = self.model(img_2)
            loss_output = self.loss_fcn(output1, output2, affine_xg_to_x)
    
        if loss_output is not None:      
            if isinstance(loss_output, dict):
                if self._weight_dict:
                    loss_output = {k: v * self._weight_dict[k] for k, v in loss_output.items() if k in self._weight_dict}
                
                losses = sum(loss for loss in loss_output.values())
                loss_output.update({'loss': losses})
            else:
                losses = loss_output[0]
                
            self.grad_scaler.scale(losses).backward()    
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            
            #write losses losses
            self._save_loss(loss_output[0])
            self._write_iter_metrics()

        self.iter += 1
        
        return True
    
    def _calculate_metrics_fromreporter(self, reporter, epoch, dict_metrics = {}):
        iter_summary = reporter.summarise_by_groups(['epoch'])
        val =  iter_summary[str(epoch)]
        for j in val.keys():
            if j in self._reporter._report_keys:
                dict_metrics[j] = val[j]
                        
        return dict_metrics
    
    def write_epoch_metrics(self, epoch = 0):
        
        if self._reporter is None:
            return None
        values = {}
        values = self._calculate_metrics_fromreporter(self._iter_tr_reporter, epoch= epoch,dict_metrics = values)
            
        if self._data_loader_eval_iter is not None:
            values = self._calculate_metrics_fromreporter(self._iter_eval_reporter, epoch= epoch, dict_metrics = values)
        
        self._reporter.update_report(values)
        self._reporter.save_reporter(path = os.path.join(self._weight_path, self._reporter.file_name), suffix = None)
        
        return values
        
def build_backbone_from_cfg(cfg):
    from models.TSViT.swinTSViT import TSViT_SingleToken
    if	cfg.model_name == 'TSViT_single_token':
        model = TSViT_SingleToken(cfg)
    
    return model

def build_model_from_cfg(cfg):
    from models.TSViT.swinTSViT import TSViT_SingleToken
    teacher = build_backbone_from_cfg(cfg)
    cfg.emb_dropout = 0.1
    student = build_backbone_from_cfg(cfg)

    return teacher, student, cfg.dim


class DINOTrainerModel(nn.Module):
    def __init__(self, config, train_data_loader, 
                reporter = None,
                sobel_filter = True,
                grad_scaler= None, 
                model_weight_path = None,
                reporter_losses = ['epoch', 'iter', 'local_loss', 'global_loss', 'loss']) -> None:
        
        super().__init__()
        student_model_dict = dict()
        teacher_model_dict = dict()
        teacher_backbone, student_backbone, embed_dim = build_model_from_cfg(config)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.embed_dim = embed_dim
        self.dino_out_dim = config.head_n_prototypes
        dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=config.head_n_prototypes,
            )
        
        self.dino_loss = DINOLossV2(self.dino_out_dim)
        
        student_model_dict["dino_head"] = dino_head()
        teacher_model_dict["dino_head"] = dino_head()
        pretrained = config.get('pretrained', None)
        
        #super().__init__(teacher)
        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        if pretrained:
            model_state_dict = torch.load(pretrained )
            self.student.load_state_dict(model_state_dict, strict=False)
            print(f"OPTIONS -- pretrained weights: loading from {pretrained}")

        for p in self.teacher.parameters():
            p.requires_grad = False

        #self.sobel_filter = sobel_filter # IIC implementation
        self._tr_data_loader = train_data_loader
        self._weight_path = model_weight_path or 'tmp'

        self.grad_scaler = grad_scaler
        self._reporter_losses = reporter_losses
        self._reporter = reporter
        #self.teacher.to(self.device)
        #self.student.to(self.device)
        
        if grad_scaler is None:
            from torch.cuda.amp import GradScaler
            self.grad_scaler = GradScaler()
        else:
            self.grad_scaler = grad_scaler
            
        
        self.set_initial_params()
        self._set_initial_reporter_params()
        
    
    @property
    def _data_loader_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = iter(self._tr_data_loader)
        return self._data_loader_iter_obj
    
    def forward(self, inputs):
        raise NotImplementedError
    
    def set_initial_params(self):
        self.iter = 0
        self.eval_iter = 0
        self._data_loader_iter_obj = None
        self._data_loader_eval_iter_obj = None


    def _set_initial_reporter_params(self):
        self._iter_tr_reporter = ReporterBase()
        self._iter_tr_reporter.set_reporter(self._reporter_losses)
        
        if self._reporter is None:
            self._reporter = ReporterBase()
            self._reporter.set_reporter(self._reporter_losses)
            self._reporter.file_name = 'reporter.json'
    
    def build_schedulers(self):

        lr = dict(
            base_value=0.004,
            final_value=1.0e-06,
            total_iters=100* len(self._tr_data_loader),
            warmup_iters=10* len(self._tr_data_loader),
            start_warmup_value=0,
        )
        wd = dict(
            base_value=0.04,
            final_value=0.4,
            total_iters=100 * len(self._tr_data_loader),
        )
        momentum = dict(
            base_value=0.992,
            final_value=1,
            total_iters=100* len(self._tr_data_loader),
        )
        teacher_temp = dict(
            base_value=0.07,
            final_value=0.07,
            total_iters=30 * len(self._tr_data_loader),
            warmup_iters=30 * len(self._tr_data_loader),
            start_warmup_value=0.04,
        )

        #self.lr_schedule = CosineScheduler(**lr)
        self.lr_schedule = self._init_lr_scheduler()
        self.wd_schedule = CosineScheduler(**wd)
        self.momentum_schedule = CosineScheduler(**momentum)
        self.teacher_temp_schedule = CosineScheduler(**teacher_temp)
        self.last_layer_lr_schedule = CosineScheduler(**lr)

        self.last_layer_lr_schedule.schedule[
            : 1 * len(self._tr_data_loader)
        ] = 0
    
    def forward_backward(self):
        """
        Run a single training iteration.
        """
        
            
        collated_imgs = next(self._data_loader_iter)
        global_crops = collated_imgs["collated_global_crops"].to(self.device)#.cuda(non_blocking=True)
        local_crops = collated_imgs["collated_local_crops"].to(self.device)#.cuda(non_blocking=True)
        teacher_temp = self.teacher_temp_schedule[self.iter]
        n_global_crops = 2
        loss_accumulator = 0
        n_local_crops = 8
        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops
        dino_loss_weight = 1
        
        #lr = self.lr_schedule[self.iter]
        #wd = self.wd_schedule[self.iter]
        mom = self.momentum_schedule[self.iter]

        teacher_temp = self.teacher_temp_schedule[self.iter]
        #last_layer_lr = self.last_layer_lr_schedule[self.iter]
        #apply_optim_scheduler(self.optimizer, lr, wd, last_layer_lr)

        # compute losses

        #self.optimizer.zero_grad(set_to_none=True)
        
        self.optimizer.zero_grad()
        
        @torch.no_grad()
        def get_teacher_output():
            x, n_global_crops_teacher = global_crops, n_global_crops
            teacher_backbone_output = self.teacher.backbone(x)
            teacher_cls_tokens = teacher_backbone_output.chunk(n_global_crops_teacher)
            teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
            #n_cls_tokens = teacher_cls_tokens.shape[0]
            teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
            
            teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                            teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                        ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
            
            self.dino_loss.update_center(teacher_cls_tokens_after_head)
            
            return teacher_dino_softmaxed_centered_list
        
        ## teacher loss
        teacher_dino_softmaxed_centered_list = get_teacher_output()
        teacher_dino_softmaxed_centered_list#.to(self.device)
        ## student outputs

        student_global_backbone_output = self.student.backbone(global_crops)
        student_global_cls_tokens_after_head = self.student.dino_head(student_global_backbone_output)#.to(self.device)

        student_local_cls_tokens = self.student.backbone(local_crops)
        student_local_cls_tokens_after_head = self.student.dino_head(student_local_cls_tokens)#.to(self.device)
        
        # local student loss
        if n_local_crops > 0:
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            
            loss_accumulator += dino_loss_weight * dino_local_crops_loss
            
        # global student loss
        loss_scales = 2
        dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

        loss_accumulator += dino_loss_weight * dino_global_crops_loss

        self.grad_scaler.scale(loss_accumulator).backward()
        
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        self.loss = loss_accumulator.detach().cpu().item()
        self.local_loss = dino_local_crops_loss.detach().cpu().item()
        self.global_loss = dino_global_crops_loss.detach().cpu().item()
        
        self._write_iter_metrics()
        update_teacher(self.student, self.teacher, mom)
        self.iter += 1
        return True

    def train_one_epoch(self):
        """
        Conduct training over one epoch.
        """
        super().train()
        self.teacher.eval()
        
        max_iter = len(self._tr_data_loader)
        pbar = tqdm(range(max_iter), leave=True, colour='blue', desc="Training")
        for _ in pbar:
            try:
                self.forward_backward()
                if self.loss is not None:
                    toshow = {}
                    for k in self._iter_tr_reporter.report.keys():
                        toshow[k] = self._iter_tr_reporter.report[k][-1]
                    
                    pbar.set_postfix(OrderedDict(toshow))
            except:
                continue
                    
    def _init_lr_scheduler(self, scheduler = 'cosine'):
        if scheduler == 'lambda':
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(self._tr_data_loader) - 1)

            self.lr_scheduler = warmup_lr_scheduler(self.optimizer, 
                                            warmup_iters, warmup_factor)
        elif scheduler == 'cosine':
            from timm.scheduler.cosine_lr import CosineLRScheduler
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(self._tr_data_loader) - 1)
            
            t_initial = int(100 * len(self._tr_data_loader) / 1)
            warmup_steps = int(1 * len(self._tr_data_loader))

            self.lr_scheduler = CosineLRScheduler(
                self.optimizer,
                t_initial=t_initial,
                # t_mul=1.,
                lr_min=float(1e-6),
                warmup_lr_init=float(1e-8),
                warmup_t=warmup_steps,
                cycle_limit=int(2),
                t_in_epochs=False,
            )
            
    def write_epoch_metrics(self, epoch = 0):
        
        if self._reporter is None:
            return None
        values = {}
        values = self._calculate_metrics_fromreporter(self._iter_tr_reporter, epoch= epoch,dict_metrics = values)
        
        self._reporter.update_report(values)
        self._reporter.save_reporter(path = self._weight_path, fn = self._reporter.file_name, suffix = None)
        
        return values
    
    def _calculate_metrics_fromreporter(self, reporter, epoch, dict_metrics = {}):
        iter_summary = reporter.summarise_by_groups(['epoch'])
        val =  iter_summary[str(epoch)]
        for j in val.keys():
            if j in self._reporter._report_keys:
                dict_metrics[j] = val[j]
                        
        return dict_metrics
    
    def _write_iter_metrics(self, evaluation = False):
        if evaluation:
            values = {k: self.__getattribute__(k) for k in self._iter_eval_reporter._report_keys}
            self._iter_eval_reporter.update_report(values)
        else:
            values = {k: self.__getattribute__(k) for k in self._iter_tr_reporter._report_keys}
            self._iter_tr_reporter.update_report(values)
            
    def _write_metrics(self):
        """
        Writes metrics using the reporter attribute if configured.
        """
        if self._reporter is None:
            return None
        
        values = {k: self.__getattribute__(k) for k in self._reporter._report_keys}
        self._reporter.update_report(values)

    def fit(self, max_epochs: int, start_from: int = 0, 
            start_saving_from =2):
        
        self.epoch = start_from
        self.build_schedulers()
        
        pbar = tqdm(range(self.epoch, max_epochs),leave=True, desc="Overall Training Progress")
        #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 100)
        start_global = 1
        for _ in pbar:
            pbar.set_description("[Epoch %d]" % (self.epoch))
            self.train_one_epoch()
            abs_step = start_global + (self.epoch - start_from) * self._data_loader_iter.__len__() + self.iter
            
            if self.epoch % 2 == 0: self.lr_scheduler.step_update(abs_step)

            self.set_initial_params()
            
            if self.epoch % 2 == 0 and self.epoch>=start_saving_from:
                new_state_dict = self.teacher.state_dict()
                outname = os.path.join(self._weight_path, f'TSViTSdino_{self.epoch}')
                torch.save(new_state_dict,  outname + '_model_params')
            epoch_metrics  = self.write_epoch_metrics(self.epoch)

            self.epoch += 1
        new_state_dict = self.teacher.state_dict()
        outname = os.path.join(self._weight_path, f'TSViTSdino_{self.epoch}')
        torch.save(new_state_dict,  outname + '_model_params')
            
            
            
class DLTrainerModel(DLBaseEngine):
    """
    Training engine for deep learning models incorporating various functionalities such as
    training, validation, logging, and gradient scaling.

    Parameters:
    ----------
    model : nn.Module
        The neural network model to train.
    train_data_loader : DataLoader
        DataLoader for training data.
    optimizer : torch.optim.Optimizer
        Optimizer used for training.
    validation_data_loader : DataLoader, optional
        DataLoader for validation data.
    reporter : Reporter, optional
        Tool to report metrics during training.
    grad_scaler : torch.cuda.amp.GradScaler, optional
        Gradient scaler for mixed precision training.
    loss_fcn : Callable, optional
        Loss function to be used during training.
    model_weight_path : str, optional
        Path to save the model weights.

    Attributes:
    ----------
    device : str
        Device to which the model and data are sent ('cuda' or 'cpu').
    """
    
    def __init__(self, model, train_data_loader,optimizer, validation_data_loader = None, 
                 reporter = None,
                 sobel_filter = True,
                 grad_scaler= None, loss_fcn = None, 
                 model_weight_path = None, weight_dict = None,
                 reporter_losses = ['epoch', 'iter','loss']) -> None:
        
        super().__init__(model)
        self.sobel_filter = sobel_filter # IIC implementation
        self._tr_data_loader = train_data_loader
        self._val_data_loader = validation_data_loader
        self.optimizer = optimizer
        self._reporter = reporter
        self._multiclass = False
        self._weight_path = model_weight_path
        self._weight_dict = weight_dict
        self.grad_scaler = grad_scaler
        self._reporter_losses = reporter_losses
        self.loss_fcn = loss_fcn
        
        if grad_scaler is None:
            from torch.cuda.amp import GradScaler
            self.grad_scaler = GradScaler()
        else:
            self.grad_scaler = grad_scaler

        
        self.model = self.model.to(self.device)
        
        optimizer_to(self.optimizer,self.device)
        self.set_initial_params()
        self._set_initial_reporter_params()
        
    @property
    def _data_loader_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = iter(self._tr_data_loader)
        return self._data_loader_iter_obj
    
    
    def set_initial_params(self):
        self.iter = 0
        self.eval_iter = 0

        self._data_loader_iter_obj = None
        self._data_loader_eval_iter_obj = None

    def _set_initial_reporter_params(self):
        self._iter_tr_reporter = ReporterBase()
        self._iter_tr_reporter.set_reporter(
                            self._reporter_losses[:2]+['train_'+i for i in self._reporter_losses[2:]])
        self._iter_eval_reporter = ReporterBase()
        self._iter_eval_reporter.set_reporter(
            self._reporter_losses[:1]+['eval_'+i for i in self._reporter_losses[1:]])
        
    
    @property
    def _data_loader_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = iter(self._tr_data_loader)
        return self._data_loader_iter_obj
    
    @property
    def _data_loader_eval_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_eval_iter_obj is None and self._val_data_loader is not None:
            self._data_loader_eval_iter_obj = iter(self._val_data_loader)
        return self._data_loader_eval_iter_obj

    
    def fit(self, max_epochs: int, start_from: int = 1, 
            save_best: bool = False, 
            checkpoint_metric: str = 'eval_loss',
            best_value: float = 100.,
            suffix_model: str = None,
            lag_best: int = None,
            start_saving_from =None):
        """
        Run the training process for a specified number of epochs.

        Parameters:
        ----------
        max_epochs : int
            Total number of epochs to train the model.
        start_from : int, optional
            The starting epoch number, useful for resuming training. Default is 0.
        """
        
        bestloss = best_value
        lastbest_epoch = 0
        #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 100)
        
        start_saving_from = start_saving_from or 0
        self.epoch = int(start_from) if start_from else 0
        suffix_model = suffix_model if suffix_model else ""
        lag_best = max_epochs if lag_best is None else lag_best
        logger = logging.getLogger(__name__)
        logger.info("Starting training from epoch {} to {}".format(self.epoch, max_epochs))
        
        pbar = tqdm(range(self.epoch, max_epochs),leave=True, desc="Overall Training Progress")
        #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 100)
        self._init_lr_scheduler() 
        start_global = 1
        
        for _ in pbar:
            pbar.set_description("[Epoch %d]" % (self.epoch))
            
            
            self.train_one_epoch()
            abs_step = start_global + (self.epoch - start_from) * self._data_loader_iter.__len__() + self.iter
            
            if self.epoch % 2 == 0: self.lr_scheduler.step_update(abs_step)
            
            self.eval_one_epoch()

            epoch_metrics  = self.write_epoch_metrics(self.epoch)
            pbar.set_postfix(OrderedDict(epoch_metrics))
            
            self.set_initial_params() # Reset or update parameters if needed per epoch
            
            if epoch_metrics[checkpoint_metric]>bestloss and save_best:
                bestloss = epoch_metrics[checkpoint_metric]
                if self.epoch>start_saving_from:
                    outname = os.path.join(self._weight_path, f'{self.model.model_name}_{self.epoch}{suffix_model}')
                    self.save_model(outname)
                    logging.info("The best model was saved at epoch: {}; {} value: {:.4f}".format(self.epoch, checkpoint_metric,bestloss))
                lastbest_epoch = self.epoch
            if lag_best < (self.epoch - lastbest_epoch):
                break
            if self.epoch % 10 == 0 and self.epoch>start_saving_from:
                outname = os.path.join(self._weight_path, f'{self.model.model_name}_{self.epoch}{suffix_model}')
                self.save_model(outname)
            self.epoch += 1
        if self._weight_path is not None:
            
            outname = os.path.join(self._weight_path, self.model.model_name+ '_last' + suffix_model)
            self.save_model(outname)
        
        
    def eval_one_epoch(self):
        """
        Evaluate the model on the validation dataset.
        """
        self.model.eval()
        max_iter = len(self._val_data_loader)
        pbar = tqdm(range(max_iter), leave=True, colour='green', desc="Evaluating")
        for _ in pbar:
            try:
                #pbar.set_description("[iter %d]" % (self.eval_iter + 1))
                self.run_eval_iter()
                
                pbar.set_postfix(OrderedDict(loss=self.eval_loss))
                
            except Exception:
                warnings.warn("Exception during Evaluation:")
                
    
    def run_eval_iter(self):
        """
        Run a single evaluation iteration.
        """
        assert not self.model.training, "Model was changed to training mode!"
        
        x, y = next(self._data_loader_eval_iter)
        y = y.to(self.device)
        x = x.to(self.device)
        y = y.to(torch.int64)
        if self.sobel_filter:
            x = sobel_filter(x)
        #with torch.cuda.amp.autocast():
        with torch.no_grad():
            output = self.model(x)
            losses = self.loss_fcn(output, y)

        self._save_class_metrics(output, y,losses, name='eval_')
        self._write_iter_metrics(evaluation= True)
        self.eval_iter +=1

    def train_one_epoch(self):
        """
        Conduct training over one epoch.
        """
        self.model.train()
        
        max_iter = len(self._tr_data_loader)
        pbar = tqdm(range(max_iter), leave=True, colour='blue', desc="Training")
        for _ in pbar:
            #try:
                endprocess = self.run_iter()
                if endprocess:
                    toshow = {}
                    for k in self._iter_tr_reporter.report.keys():
                        toshow[k] = self._iter_tr_reporter.report[k][-1]
                    
                    pbar.set_postfix(OrderedDict(toshow))
            #except Exception:
            #    continue
            #    warnings.warn("Exception during training:")
                
    def run_iter(self):
        """
        Run a single training iteration.
        """
        
        assert self.model.training, "Model was changed to eval mode!"
        
        x, y = next(self._data_loader_iter)
        y = y.to(self.device)
        x = x.to(self.device)
        y = y.to(torch.int64)
        if self.sobel_filter:
            x = sobel_filter(x)
            
        self.optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            output = self.model(x)
            loss_output = self.loss_fcn(output, y)
            
        if loss_output is not None:      
            

            # Unscale before clipping
            #self.grad_scaler.unscale_(self.optimizer)
            # Gradient clipping (added)
            #torch.nn.utils.clip_grad_norm_(
            #    self.model.parameters(),
            #    max_norm=1.0,  # Adjust based on your needs
            #    norm_type=2.0,
            #    error_if_nonfinite=True
            #)
            self.grad_scaler.scale(loss_output).backward()    
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
                            
            #write losses losses
            #self._save_loss(loss_output)
            self._save_class_metrics(output, y,loss_output, name='train_')
            self._write_iter_metrics()

            #self.lr_scheduler.step()
        self.iter += 1
        
        return True
        
    def _save_class_metrics(self, pred, real, lossvalue, name = ''):
        metrics_dict = get_mean_metrics(
                    logits=pred, labels=real, n_classes=self._nclasses, loss=lossvalue, name = name)
        
        for k,v in metrics_dict.items():
            exec('self.'+k + '=v')
                        
    
    def _calculate_metrics_fromreporter(self, reporter, epoch, dict_metrics = {}):
        iter_summary = reporter.summarise_by_groups(['epoch'])
        val =  iter_summary[str(epoch)]
        for j in val.keys():
            if j in self._reporter._report_keys:
                dict_metrics[j] = val[j]
                        
        return dict_metrics
    
    def write_epoch_metrics(self, epoch = 0):
        
        if self._reporter is None:
            return None
        values = {}
        values = self._calculate_metrics_fromreporter(self._iter_tr_reporter, epoch= epoch,dict_metrics = values)
            
        if self._data_loader_eval_iter is not None:
            values = self._calculate_metrics_fromreporter(self._iter_eval_reporter, epoch= epoch, dict_metrics = values)
        
        self._reporter.update_report(values)
        self._reporter.save_reporter(path = os.path.join(self._weight_path),fn = self._reporter.file_name, suffix = None)
        
        return values


           
from models.loss_functions import FocalLoss, MaskedFocalLoss

class ConvSegHeadvold(nn.Module):
    def __init__(self, dim, patch_size, num_classes):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
        # Reduce channel dimension then upsample
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim//2), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dim//2, dim//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim//4), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dim//4, num_classes, kernel_size=1),
            nn.Softmax2d()
        )

    def forward(self, tokens, H, W):
        B, NP, D = tokens.shape
        Hp, Wp = H//self.patch_size, W//self.patch_size
        x = tokens.transpose(1,2).view(B, D, Hp, Wp)   # (B, D, Hp, Wp)
        x = self.decoder(x)                            # (B, C, H, W)
        
        x = F.interpolate(x, size=H, mode="bilinear")
        
        return x


class ConvSegHead(nn.Module):
    
    def __init__(self, dim: int,  
                patch_size: int, 
                num_classes: int,  
                embedding_dim: int = 256, 
                dropout_prob: float = 0.1): 
        
        super().__init__()

        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        
        self.norm = nn.LayerNorm(dim)
        self.linear_proj = nn.Linear(dim, embedding_dim)
        self.dropout = nn.Dropout2d(dropout_prob)
        self.cls_seg = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, tokens, H: int, W: int):
        B, NP, D = tokens.shape
        x = self.norm(tokens)
        Hp = H // self.patch_size
        Wp = W // self.patch_size
        x = self.linear_proj(x)
        x = x.view(B, Hp, Wp, self.embedding_dim)
        
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.dropout(x)
        x = self.cls_seg(x)
        x = F.interpolate(
                x,
                size=(H, W),        # Target output size H, W
                mode='bilinear',
                align_corners=False
            )
        
        return x


class PatchExpandSegHeadvold(nn.Module):
    def __init__(self, dim, patch_size, num_classes):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
        # Project D → patch_size² * num_classes
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim*4, patch_size*patch_size * num_classes),
        )

    def forward(self, tokens, H, W):
        """
        tokens: (B, Hp*Wp, D)
        H, W: original image height/width (must be divisible by patch_size)
        """
        B, NP, D = tokens.shape
        Hp = H // self.patch_size
        Wp = W // self.patch_size

        # 1) Project to per-patch logits
        x = self.to_logits(tokens.reshape(-1, D))                                      # (B, NP, ps²·C)
        # 2) Rearrange → (B, C, H, W)
        x = x.reshape(B, self.num_classes, Hp*Wp, self.patch_size**2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        
        return x

class PatchExpandSegHead(nn.Module):
    def __init__(self, dim, patch_size, num_classes):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
        # Project D → patch_size² * num_classes
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim*4, patch_size*patch_size * num_classes),
        )

    def forward(self, tokens, H, W):
        """
        tokens: (B, Hp*Wp, D)
        H, W: original image height/width (must be divisible by patch_size)
        """
        B, NP, D = tokens.shape
        Hp = H // self.patch_size
        Wp = W // self.patch_size

        # 1) Project to per-patch logits
        x = self.to_logits(tokens)                                      # (B, NP, ps²·C)
        # 2) Rearrange

        x = x.view(B, Hp, Wp, self.patch_size, self.patch_size, self.num_classes) 
        
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()     # (B, C, Hp, ps, Wp, ps)
        # Reshape to final image dimensions
        x = x.view(B, self.num_classes, Hp * self.patch_size, Wp * self.patch_size) 
 
        return x


class DINOTrainerSegmentationModel(nn.Module):
    def __init__(self, config, train_data_loader,
                 eval_data_loader = None,
                 loss_fn = None,
                reporter = False,
                optimizer = None,
                grad_scaler= None, 
                model_weight_path = None,
                backbone_weights = None,
                froze_backbone = True,
                reporter_losses = ['epoch', 'iter', 'loss']) -> None:
        
        super().__init__()
        self.froze_backbone = froze_backbone
        model_dict = dict()
        model_dict["backbone"] = build_backbone_from_cfg(config)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._nclasses = config.n_classes + 1 ## background

        if loss_fn is None: 
            loss_fn = FocalLoss(reduction='mean', gamma= 1)

        self.loss_fcn = loss_fn
        
        if backbone_weights:
            model_state_dict = torch.load(backbone_weights, map_location=torch.device('cpu') )
            model_dict["backbone"].load_state_dict(model_state_dict, strict = False)
            print('** Backbone weights loaded **')
            if froze_backbone:
                for p in model_dict["backbone"].parameters():
                    p.requires_grad = False
        
        
        if config.head_type == 'conv':
            model_dict["head"] = partial(
                    ConvSegHead,
                    dim=config.dim,
                    patch_size = config.patch_size, 
                    num_classes = self._nclasses
                )()
        else:
            model_dict["head"] = partial(
                    PatchExpandSegHead,
                    dim=config.dim,
                    patch_size = config.patch_size, 
                    num_classes = self._nclasses
                )()
        #super().__init__(teacher)
        self.model = nn.ModuleDict(model_dict)
        
        self._tr_data_loader = train_data_loader
        self._val_data_loader = eval_data_loader
        
        self._weight_path = model_weight_path or 'tmp'

        self.grad_scaler = grad_scaler
        self._reporter_losses = reporter_losses
        self._reporter = reporter
        self.model.to(self.device)
        #self.student.to(self.device)

        if grad_scaler is None:
            from torch.cuda.amp import GradScaler
            self.grad_scaler = GradScaler()
        else:
            self.grad_scaler = grad_scaler
        
        self.set_initial_params()
        self._set_initial_reporter_params()
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                    self.model.head.parameters(),      
                    lr=1e-4,
                    weight_decay=0,)
        else:
            self.optimizer = optimizer

        
        self.model_name = 'dino_segmentation'
        
    @property
    def _data_loader_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = iter(self._tr_data_loader)
        return self._data_loader_iter_obj
    
    @property
    def _data_loader_eval_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_eval_iter_obj is None and self._val_data_loader is not None:
            self._data_loader_eval_iter_obj = iter(self._val_data_loader)
        return self._data_loader_eval_iter_obj

    
    def forward(self, inputs):
        raise NotImplementedError
    
    def set_initial_params(self):
        self.iter = 0
        self.eval_iter = 0
        self._data_loader_iter_obj = None
        self._data_loader_eval_iter_obj = None


    def _set_initial_reporter_params(self):
        def set_reporter(reporter_keys, training = None):
            preffix = ''
            if training is not None:
                preffix = 'train_' if training else 'eval_'
                
            reporter =  ReporterBase()
            reporter.set_reporter(
                            reporter_keys[:2]+[preffix+i for i in reporter_keys[2:]])
            return reporter
            
        self._iter_tr_reporter = set_reporter(self._reporter_losses, training= True)
        
        if self._val_data_loader is not None:
            self._iter_eval_reporter = set_reporter(self._reporter_losses, training= False)
            
        if self._reporter is None:
            self._reporter = set_reporter(self._reporter_losses, training= None)
            self._reporter.file_name = 'reporter.json'
    
    def build_schedulers(self):

        self.lr_schedule = self._init_lr_scheduler()
    
    @torch.no_grad()
    def intermediate_spatial_features(self, img):
        #x, masks = next(dino_seg_model._data_loader_iter)

        B, T, C, H, W = img.shape
        tmp_tr = []
        for b in range(B):
            n_img = img[b].unsqueeze(dim = 0)
            n_img = self.model.backbone.temporal_transform(n_img)
            if n_img is None: return None
            tmp_tr.append(n_img)
            
        x_c = torch.concat(tmp_tr, dim = 0)#self.temporal_transform(x)
        new_num_patches = (H // self.model.backbone.patch_size) * (W // self.model.backbone.patch_size)
        x_c = x_c.view(len(tmp_tr), new_num_patches, self.model.backbone.dim)

        spatial_features = self.model.backbone.spatial_transform(x_c, H, W)
        
        return spatial_features
    
    def forward_backward(self):
        """
        Run a single training iteration.
        """
        
            
        imgs, masks = next(self._data_loader_iter)
        
        masks = masks.to(torch.int64)
        imgs, masks = imgs.to(self.device), masks.to(self.device)
        
        self.optimizer.zero_grad()
        
        #with torch.no_grad():
        backbone_feats = self.intermediate_spatial_features(imgs)
        
        preds = self.model.head(backbone_feats, 
                        self.model.backbone.image_size, self.model.backbone.image_size)
        ## teacher loss
        global_loss = self.loss_fcn(preds, masks)
        
        self.grad_scaler.scale(global_loss).backward()
        
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        self._save_class_metrics(preds, masks, global_loss, name='train_')
        
        self.loss = global_loss.detach().cpu().item()
        
        self._write_iter_metrics()
        
        self.iter += 1
        return True
    
    def _save_class_metrics(self, pred, real, lossvalue, name = ''):
        metrics_dict = get_mean_metrics(
                    logits=pred, labels=real, n_classes=self._nclasses, loss=lossvalue, name = name)
        
        for k,v in metrics_dict.items():
            exec('self.'+k + '=v')
            
    def train_one_epoch(self):
        """
        Conduct training over one epoch.
        """
        super().train()
        #self.teacher.eval()
        self.model.train()
        max_iter = len(self._tr_data_loader)
        pbar = tqdm(range(max_iter), leave=True, colour='blue', desc="Training")
        for _ in pbar:
            #try:
                self.forward_backward()
                if self.loss is not None:
                    toshow = {}
                    for k in self._iter_tr_reporter.report.keys():
                        toshow[k] = self._iter_tr_reporter.report[k][-1]
                    
                    pbar.set_postfix(OrderedDict(toshow))
                    
    def _init_lr_scheduler(self, scheduler = 'cosine', lr_base = 0.004):
        if scheduler == 'lambda':
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(self._tr_data_loader) - 1)

            self.lr_scheduler = warmup_lr_scheduler(self.optimizer, 
                                            warmup_iters, warmup_factor)
        elif scheduler == 'cosine':
            from timm.scheduler.cosine_lr import CosineLRScheduler
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(self._tr_data_loader) - 1)
            
            t_initial = int(100 * len(self._tr_data_loader) / 1)
            warmup_steps = int(1 * len(self._tr_data_loader))

            self.lr_scheduler = CosineLRScheduler(
                self.optimizer,
                t_initial=t_initial,
                # t_mul=1.,
                lr_min=lr_base,
                warmup_lr_init=float(1e-8),
                warmup_t=warmup_steps,
                cycle_limit=int(2),
                t_in_epochs=False,
            )
            
    def write_epoch_metrics(self, epoch = 0):
        
        if self._reporter is None:
            return None
        values = {}
        values = self._calculate_metrics_fromreporter(self._iter_tr_reporter, epoch= epoch,dict_metrics = values)
        if self._data_loader_eval_iter is not None:
            values = self._calculate_metrics_fromreporter(self._iter_eval_reporter, epoch= epoch, dict_metrics = values)
        
        self._reporter.update_report(values)
        self._reporter.save_reporter(path = self._weight_path, fn = self._reporter.file_name, suffix = None)
        
        return values
    
    def _calculate_metrics_fromreporter(self, reporter, epoch, dict_metrics = {}):
        iter_summary = reporter.summarise_by_groups(['epoch'])
        val =  iter_summary[str(epoch)]
        for j in val.keys():
            if j in self._reporter._report_keys:
                dict_metrics[j] = val[j]
                        
        return dict_metrics
    
    def _write_iter_metrics(self, evaluation = False):
        if evaluation:
            values = {k: self.__getattribute__(k) for k in self._iter_eval_reporter._report_keys}
            self._iter_eval_reporter.update_report(values)
        else:
            values = {k: self.__getattribute__(k) for k in self._iter_tr_reporter._report_keys}
            self._iter_tr_reporter.update_report(values)
            
    def _write_metrics(self):
        """
        Writes metrics using the reporter attribute if configured.
        """
        if self._reporter is None:
            return None
        
        values = {k: self.__getattribute__(k) for k in self._reporter._report_keys}
        self._reporter.update_report(values)

    def fit(self, max_epochs: int, start_from: int = 0, 
            checkpoint_metric: str = 'eval_loss',
            bestloss: float = 100.,
            save_best: bool = True,
            lag_best: int = 20,
            start_saving_from =2):
        
        self.epoch = start_from
        self.build_schedulers()
        
        pbar = tqdm(range(self.epoch, max_epochs),leave=True, desc="Overall Training Progress")
        #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 100)
        start_global = 1
        for _ in pbar:
            pbar.set_description("[Epoch %d]" % (self.epoch))
            self.train_one_epoch()
            abs_step = start_global + (self.epoch - start_from) * self._data_loader_iter.__len__() + self.iter
            
            if self.epoch % 2 == 0: 
                self.lr_scheduler.step_update(abs_step)
            self.eval_one_epoch()
        
            epoch_metrics  = self.write_epoch_metrics(self.epoch)
            
            self.set_initial_params()
            
            outname = os.path.join(self._weight_path, f'{self.model_name}_{self.epoch}')
            print(epoch_metrics[checkpoint_metric])

            if epoch_metrics[checkpoint_metric]>bestloss and save_best:
                bestloss = epoch_metrics[checkpoint_metric]
                if self.epoch>start_saving_from:
                    self.save_model(outname)
                    logging.info("The best model was saved at epoch: {}; {} value: {:.4f}".format(self.epoch, checkpoint_metric,bestloss))
                lastbest_epoch = self.epoch
            #if lag_best < (self.epoch - lastbest_epoch):
            #    break
            if self.epoch % 10 == 0 and self.epoch>start_saving_from:

                self.save_model(outname)
                
            self.epoch += 1
    
    def save_model(self, path):
        """
        Saves the model and optimizer states to the specified path.

        Parameters
        ----------
        path : str
            Path to the directory where the model and optimizer states will be saved.

        Raises
        ------
        AssertionError
            If the specified path does not exist.
        """
        #assert os.path.exists(path), "The specified path does not exist."
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(),  path + '_scheduler_params')    
        torch.save(self.model.state_dict(),  path + '_model_params')
        torch.save(self.optimizer.state_dict(), path + '_optimizer_params')
        
        pathm = path + "_scaler_params"
        if self.grad_scaler:
            torch.save(self.grad_scaler.state_dict(), pathm)

    def load_weights(self, path_dict):
        """
        Loads the model, optimizer, and gradient scaler states from the specified paths.

        Parameters
        ----------
        path_dict : dict
            Directory that contains the paths each file from which the model state will be loaded.
        Raises
        ------
        FileNotFoundError
            If any of the specified files does not exist.
        """
        model_path = path_dict.get('model_path', None)
        model_state_dict = path_dict.get('model_state_dict', None)
        optimizer_path = path_dict.get('optimizer_path', None)
        scaler_path = path_dict.get('scaler_path', None)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model path {model_path} does not exist.")
        model_state_dict = torch.load(model_path, map_location=torch.device('cpu') )
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)

        if optimizer_path and os.path.exists(optimizer_path):
            optimizer_state_dict = torch.load(optimizer_path, map_location=torch.device(self.device))
            self.optimizer.load_state_dict(optimizer_state_dict)
        elif optimizer_path:
            raise FileNotFoundError(f"The optimizer path {optimizer_path} does not exist.")

        if scaler_path and os.path.exists(scaler_path):
            scaler_state_dict = torch.load(scaler_path)
            self.grad_scaler.load_state_dict(scaler_state_dict)
        elif scaler_path:
            raise FileNotFoundError(f"The scaler path {scaler_path} does not exist.")

        print("Model and other components (if specified) loaded successfully.")

            
    def eval_one_epoch(self):
        """
        Evaluate the model on the validation dataset.
        """
        self.model.eval()
        max_iter = len(self._val_data_loader)
        pbar = tqdm(range(max_iter), leave=True, colour='green', desc="Evaluating")
        for _ in pbar:
            try:
                #pbar.set_description("[iter %d]" % (self.eval_iter + 1))
                self.run_eval_iter()
                
                pbar.set_postfix(OrderedDict(loss=self.eval_loss))
                
            except Exception:
                warnings.warn("Exception during Evaluation:")
                
    
    def run_eval_iter(self):
        """
        Run a single evaluation iteration.
        """
        assert not self.model.training, "Model was changed to training mode!"
        
        x, y = next(self._data_loader_eval_iter)
        y = y.to(self.device)
        x = x.to(self.device)
        y = y.to(torch.int64)

        #with torch.cuda.amp.autocast():
        backbone_feats = self.intermediate_spatial_features(x)
        
        with torch.no_grad():
            preds = self.model.head(backbone_feats, 
                            self.model.backbone.image_size, self.model.backbone.image_size)
            losses = self.loss_fcn(preds, y)

        self._save_class_metrics(preds, y,losses, name='eval_')
        self.eval_loss = losses.detach().cpu().item()
        self._write_iter_metrics(evaluation= True)
        self.eval_iter +=1
