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
from models.TSViT.swinTSViT import DINOTSViT
import torch.nn.functional as F
import torch.optim as optim

from torch.cuda.amp import GradScaler

from models.metrics.numpy_metrics import get_classification_metrics

def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms

def build_model_from_cfg(config):
    
    student = DINOTSViT(config.MODEL) # Assuming config.MODEL holds backbone config
    teacher = DINOTSViT(config.MODEL)
    embed_dim = config.MODEL['dim']
    return teacher, student, embed_dim


class DINOTrainerModel(nn.Module):
    
    def __init__(self, config, train_data_loader, 
            grad_scaler= None,
            reporter = None,
            model_weight_path = None) -> None:
        
        super().__init__()
        ## save configuration
        self.config = config
        self._reporter_losses = ['epoch', 'iter_in_epoch', 'global_iter', 'loss', 'global_loss', 'local_loss','current_lr']
        self._reporter = reporter   
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        teacher_backbone, student_backbone, embed_dim = build_model_from_cfg(config)
        self.embed_dim = embed_dim
        self.dino_out_dim = config.DINO.head_n_prototypes
        
        # DINO Head
        dino_head_args = dict(
            in_dim=embed_dim,
            out_dim=self.dino_out_dim,
            hidden_dim=config.DINO.get("dino_hidden_dim", 2048),
            nlayers=config.DINO.get("dino_n_layers", 3),
            bottleneck_dim = config.DINO.get("dino_bottleneck_dim", 256)
        )
        dino_head = partial(
                DINOHead,**dino_head_args
            )
        
        #models
        self.student = nn.ModuleDict({"backbone": student_backbone, "dino_head": dino_head()}).to(self.device)
        self.teacher = nn.ModuleDict({"backbone": teacher_backbone, "dino_head": dino_head()}).to(self.device)
        
        # Initialize teacher with student weights
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
        
        # Teacher does not require gradients
        for p in self.teacher.parameters():
            p.requires_grad = False

        #
        self.dino_loss_fn = DINOLossV2(
            self.dino_out_dim,
            student_temp=config.TRAIN.get("dino_student_temp", 0.1),
            center_momentum=config.TRAIN.get("dino_center_momentum", 0.9) # DINOv1 used 0.9, DINOv2 0.996
        ).to(self.device)
        
        ## dataset
        self._tr_data_loader = train_data_loader
        self.n_global_crops = config.DATA.get("n_global_crops", 2)
        self.n_local_crops = config.DATA.get("n_local_crops", 8) # DINOv2 uses more, e.g., 8-10

        ## weigths output path
        self._weight_path = model_weight_path or 'tmp_dino_weights'
        os.makedirs(self._weight_path, exist_ok=True)
        
        # gradient
        self.grad_scaler = grad_scaler if grad_scaler is not None else GradScaler()
        self.clip_grad = config.TRAIN.get("clip_grad", 3.0) # Default DINOv2 clip_grad

        self._set_initial_params()
        self._set_initial_reporter_params()
        self._setup_optimizer() # Call optimizer setup
        self.build_schedulers() #
        
    def _set_initial_params(self):
        
        self.iter_in_epoch = 0
        self.global_iter = 0
        self.epoch = 0
        self._data_loader_iter_obj = None
        
    
    def build_schedulers(self):
        cfg_sched = self.config.SCHEDULER
        total_iters = self.config.TRAIN.max_epochs * len(self._tr_data_loader)

        lr_params = dict(
            base_value=cfg_sched.lr.get('base_value', 5e-4), # Peak LR
            final_value=cfg_sched.lr.get('final_value', 1e-6),
            total_iters=total_iters,
            warmup_iters=cfg_sched.lr.get('warmup_epochs', 10) * len(self._tr_data_loader),
            start_warmup_value=cfg_sched.lr.get('start_warmup_value', 1e-8)
        )
        wd_params = dict(
            base_value=cfg_sched.wd.get('base_value', 0.04),
            final_value=cfg_sched.wd.get('final_value', 0.4),
            total_iters=total_iters,
            
            warmup_iters=0, start_warmup_value=cfg_sched.wd.get('base_value', 0.04)
        )
        momentum_params = dict( # For teacher EMA
            base_value=cfg_sched.momentum.get('base_value', 0.996), 
            final_value=cfg_sched.momentum.get('final_value', 1.0),
            total_iters=total_iters,
            warmup_iters=0, start_warmup_value=cfg_sched.momentum.get('base_value', 0.996)
        )
        teacher_temp_params = dict(
            base_value=cfg_sched.teacher_temp.get('base_value', 0.07),
            final_value=cfg_sched.teacher_temp.get('final_value', 0.04), # DINOv2 often decreases temp slightly
            total_iters=30 * len(self._tr_data_loader),
            warmup_iters=30 * len(self._tr_data_loader),
            start_warmup_value=cfg_sched.teacher_temp.get('start_warmup_value', 0.04),
        )

        self.lr_schedule = CosineScheduler(**lr_params)
        self.wd_schedule = CosineScheduler(**wd_params) # For params with WD
        self.momentum_schedule = CosineScheduler(**momentum_params)
        self.teacher_temp_schedule = CosineScheduler(**teacher_temp_params)
        self.last_layer_lr_schedule = CosineScheduler(**lr_params)
        
    def _set_initial_reporter_params(self):
        self._iter_tr_reporter = ReporterBase()
        self._iter_tr_reporter.set_reporter(self._reporter_losses)
        
        if self._reporter is None:
            self._reporter = ReporterBase()
            self._reporter.set_reporter(self._reporter_losses)
            self._reporter.file_name = 'reporter.json'
            
    def _setup_optimizer(self):
        param_groups = []
        base_lr = self.config.SCHEDULER.lr.base_value
        base_wd = self.config.SCHEDULER.wd.base_value
        
        param_groups.append({
            'params': self.student.dino_head.parameters(),
            'lr_scale': self.config.TRAIN.get("dino_head_lr_scale", 1.0),
            'weight_decay': 0.0
        })
        
        # Group 2: Backbone weights (excluding LayerNorm/Bias)
        no_decay_keys = ["bias", ".norm.weight", ".norm_final.weight", 
                        "space_pos_embedding", "cls_token_spatial", "cls_temporal_token"] # Add other no_decay params
        
        backbone_weights, backbone_no_decay = [], []
        for name, param in self.student.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if any(key in name for key in no_decay_keys) or len(param.shape) == 1:
                backbone_no_decay.append(param)
            else:
                backbone_weights.append(param)
                
        
        param_groups.append({"params": backbone_weights, "weight_decay": base_wd, "lr_scale": 1.0})
        param_groups.append({"params": backbone_no_decay, "weight_decay": 0.0, "lr_scale": 1.0})
        self.optimizer = optim.AdamW(param_groups, lr=base_lr, weight_decay=base_wd) # WD for group 2 applied here too
        print(f"Optimizer: AdamW with {len(param_groups)} parameter groups.")
        
    @property
    def _data_loader_iter(self):
        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = iter(self._tr_data_loader)
        return self._data_loader_iter_obj
        
    def _apply_scheduler(self):
        
        current_lr_base = self.lr_schedule[self.global_iter]
        current_wd_base = self.wd_schedule[self.global_iter]
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            lr_scale = param_group.get('lr_scale', 1)
            param_group['lr'] = current_lr_base *lr_scale
            
            if param_group.get("weight_decay", "default_wd_marker") == 0.0:
                param_group['weight_decay'] = 0.0
            
            elif "weight_decay" in param_group: # If WD is set for this group (e.g. base_wd for weights)
                param_group['weight_decay'] = current_wd_base
                
        self.current_lr = self.optimizer.param_groups[0]['lr']
    
    def _write_iter_metrics(self):
        values = {k: self.__getattribute__(k) for k in self._iter_tr_reporter._report_keys}
        self._iter_tr_reporter.update_report(values)
    
    def _calculate_metrics_fromreporter(self, reporter, epoch, dict_metrics = {}):
        iter_summary = reporter.summarise_by_groups(['epoch'])
        val =  iter_summary[str(epoch)]
        for j in val.keys():
            if j in self._reporter._report_keys:
                dict_metrics[j] = val[j]
                        
        return dict_metrics
    
    def forward(self, inputs):
        raise NotImplementedError
    
    def forward_backward(self):
        n_global_crops = 2
        loss_accumulator = 0
        n_local_crops = 8
        
        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops
        
        self.student.train()
        self.teacher.eval() 
        dino_loss_weight = 1
        self._apply_scheduler()
        self.optimizer.zero_grad(set_to_none=True)
        
        collated_imgs = next(self._data_loader_iter)
#        except:
#            self._data_loader_iter = iter(self._tr_data_loader)
#            collated_imgs = next(self._data_loader_iter)
        
        global_crops = collated_imgs["collated_global_crops"].to(self.device, non_blocking=True)#.cuda(non_blocking=True)
        local_crops = collated_imgs["collated_local_crops"].to(self.device, non_blocking=True)#.cuda(non_blocking=True)
        nbatches = global_crops.shape[0]//n_global_crops
        with torch.no_grad():
            teacher_globalbackbone_output, _ = self.teacher.backbone(global_crops, return_patch_tokens=False)
            # --- Check Student Output Shapes using B_eff ---

            teacher_globalbackbone_output = teacher_globalbackbone_output.chunk(n_global_crops)
            teacher_globalbackbone_output = torch.cat((teacher_globalbackbone_output[1], teacher_globalbackbone_output[0]))
            B_eff = teacher_globalbackbone_output.shape[0] // n_global_crops
            current_teacher_temp = self.teacher_temp_schedule[self.global_iter]
            teacher_global_dino_out = self.teacher.dino_head(teacher_globalbackbone_output)
            teacher_dino_softmaxed_centered_list = self.dino_loss_fn.softmax_center_teacher(
                            teacher_global_dino_out, teacher_temp=current_teacher_temp
                        ).view(n_global_crops, -1, *teacher_global_dino_out.shape[1:])
            
            self.dino_loss_fn.update_center(teacher_global_dino_out)
            
        with torch.cuda.amp.autocast(enabled=False):#self.grad_scaler.is_enabled()): ##
            loss_accumulator = 0
            student_global_backbone_output, _ = self.student.backbone(global_crops)
            student_global_cls_tokens_after_head = self.student.dino_head(student_global_backbone_output)#.to(self.device)

            student_local_cls_tokens, skipped_times = self.student.backbone(local_crops)
            student_local_cls_tokens_after_head = self.student.dino_head(student_local_cls_tokens)#.to(self.device)
            notvalid_set = set(skipped_times)
            pos = [[i for i in range(B_eff) if B_eff * j + i not in notvalid_set] for j in range(n_local_crops)]
            
            if not student_local_cls_tokens.shape[0] == B_eff * n_local_crops: 
                #print(f"Student local output shape mismatch: expected {B_eff*n_local_crops}, got {student_local_cls_tokens.shape[0]}")
                #print(student_local_cls_tokens.shape)
                #print(pos)
                #print(teacher_dino_softmaxed_centered_list.shape)
                student_local_cls_tokens_after_headc = [
                    student_local_cls_tokens_after_head[[j * B_eff + i - sum(1 for x in skipped_times if x < j * B_eff + i) for i in crop_indices]]
                    for j, crop_indices in enumerate(pos)
                    if crop_indices  # Skip empty crops if needed
                ]
            else:
                student_local_cls_tokens_after_headc = student_local_cls_tokens_after_head.chunk(n_local_crops)
            # local student loss
            #try:
            dino_local_crops_loss = self.dino_loss_fn(
                student_output_list=student_local_cls_tokens_after_headc,
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            #print('local after backbone', student_local_cls_tokens.shape)

            #except:
            #    print('local after backbone', student_local_cls_tokens.shape)
            #    print('local after ', student_local_cls_tokens_after_head.shape)
            #    print('local crop', local_crops.shape)
            #    print('global', global_crops.shape)
            #    print(teacher_dino_softmaxed_centered_list.shape)


            loss_accumulator = dino_loss_weight * dino_local_crops_loss
                
            # global student loss
            
            #student_global_views = student_global_cls_tokens_after_head.chunk(self.n_global_crops) # List of 2 tensors: [(1,P), (1,P)]
            #teacher_global_views_sm_cent = teacher_dino_softmaxed_centered_list # Already (2, 1, P), can be chunked or indexed

            #loss_g1_t2 = self.dino_loss_fn(student_output_list=[student_global_views[0]],
            #                            teacher_out_softmaxed_centered_list=[teacher_global_views_sm_cent[1]])
            #loss_g2_t1 = self.dino_loss_fn(student_output_list=[student_global_views[1]],
            #                            teacher_out_softmaxed_centered_list=[teacher_global_views_sm_cent[0]])
            #dino_global_crops_loss = (loss_g1_t2 + loss_g2_t1) / 2.0 # Average the two cross-view losses
            #dino_global_crops_loss = dino_global_crops_loss / (n_global_crops_loss_terms + n_local_crops_loss_terms) # Then normalize by total terms
            loss_scales = 2
            dino_global_crops_loss = (
                self.dino_loss_fn(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            loss_accumulator += dino_loss_weight * dino_global_crops_loss
    
    # Backward pass and optimization
    
        #self.grad_scaler.scale(loss_accumulator).backward()
        loss_accumulator.backward()
        if self.clip_grad:
            clip_gradients(self.student, self.clip_grad)

        #if self.clip_grad > 0:
        #    self.grad_scaler.unscale_(self.optimizer) # Unscale before clipping
        #    nn.utils.clip_grad_norm_(self.student.parameters(), self.clip_grad)
        
        #self.grad_scaler.step(self.optimizer)
        self.optimizer.step()
        #self.grad_scaler.update()
        
        self.loss = loss_accumulator.detach().cpu().item()
        self.local_loss = dino_local_crops_loss.detach().cpu().item()
        self.global_loss = dino_global_crops_loss.detach().cpu().item()
        
        self._write_iter_metrics()
        # Teacher EMA update
        current_teacher_momentum = self.momentum_schedule[self.global_iter]
        update_teacher(self.student, self.teacher, current_teacher_momentum)
        self.iter_in_epoch += 1
        self.global_iter += 1
        
        return True

    def train_one_epoch(self):
        """
        Conduct training over one epoch.
        """
        super().train()
        self.teacher.eval()
        #torch.autograd.set_detect_anomaly(True)
        self.iter_in_epoch = 0
        
        if self._data_loader_iter_obj is not None: # new iterator for new epoch
            self._data_loader_iter_obj = None
            
        max_iter_epoch = len(self._tr_data_loader)
        pbar = tqdm(range(max_iter_epoch), leave=True, colour='green', desc=f"Epoch {self.epoch} Training")
        
        for _ in pbar:
            try: 
                self.forward_backward()
                if hasattr(self, 'loss') and self.loss is not None:
                    toshow = {
                    'iter': f"{self.iter_in_epoch}",
                    'loss': f"{self.loss:.4f}",
                    'g_loss': f"{self.global_loss:.4f}",
                    'l_loss': f"{self.local_loss:.4f}",
                    'lr': f"{self.current_lr:.2e}",
                    'mom': f"{self.momentum_schedule[self.global_iter-1]:.4f}", # iter was just incremented
                    'tmp': f"{self.teacher_temp_schedule[self.global_iter-1]:.3f}"
                 }
                
                    pbar.set_postfix(OrderedDict(toshow))
            except:
                continue
            

    def save_checkpoint(self, filename_suffix="_"):
        checkpoint = {
            'epoch': self.epoch,
            'global_iter': self.global_iter,
            'student_state_dict': self.student.state_dict(),
            'teacher_state_dict': self.teacher.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'grad_scaler_state_dict': self.grad_scaler.state_dict(),
            'dino_loss_center': self.dino_loss_fn.center,
        }
        filepath = os.path.join(self._weight_path, f'TSViTSdino_checkpoint_{filename_suffix}ep{self.epoch}.pth')
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        if not os.path.exists(filepath):
            print(f"Checkpoint file not found: {filepath}")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device, weights_only = False)
        
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.teacher.load_state_dict(checkpoint['teacher_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])
        #self.dino_loss_fn.center = checkpoint['dino_loss_center'].to(self.device)
        
        self.epoch = 0#checkpoint['epoch']
        self.global_iter = 0#checkpoint['global_iter']
        
        # Note: Restoring CosineScheduler states might require re-initializing them
        # up to the loaded global_iter if they don't have a simple state_dict.
        # For CosineScheduler as implemented, just starting from global_iter is fine.
        print(f"Checkpoint loaded from {filepath}. Resuming from epoch {self.epoch}, global_iter {self.global_iter}.")
        return True
    
    def write_epoch_metrics(self, epoch = 0):
        
        if self._reporter is None:
            return None
        values = {}
        values = self._calculate_metrics_fromreporter(self._iter_tr_reporter, epoch= epoch,dict_metrics = values)
        
        self._reporter.update_report(values)
        self._reporter.save_reporter(path = self._weight_path, fn = self._reporter.file_name, suffix = None)
        
        return values
    
    def fit(self, max_epochs: int, 
            start_saving_from =2, resume_from_checkpoint: str = None, save_every_epochs:int = 5):
        
        self.epoch = 0
        self.global_iter = 0
        if resume_from_checkpoint:
            if not self.load_checkpoint(resume_from_checkpoint):
                print(f" Couldn't resume from checkpoint, starting from scratch" )

        pbar = tqdm(range(self.epoch, max_epochs), initial=self.epoch, leave=True, desc="Overall Training Progress")
        for current_epoch_num in pbar:
            
            self.epoch = current_epoch_num
            self.train_one_epoch()
            self.write_epoch_metrics(self.epoch)
            
            if self.epoch >= start_saving_from and (self.epoch % save_every_epochs == 0 or self.epoch == max_epochs -1):
                self.save_checkpoint(filename_suffix=f"iter{self.global_iter}_")