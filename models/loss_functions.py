import torch.nn.functional as F
from torch import nn
import torch
import numpy as np


import torch.distributed as dist




from datasets.transforms.tensor_transforms import perform_affine_tf
from sys import float_info

from torch.autograd import Variable

EPS = float_info.epsilon
# taken from https://github.com/sebastiani/IIC/blob/master/code/utils/segmentation/IID_losses.py
def IID_segmentation_loss(x1_outs, x2_outs, all_affine2_to_1=None,
                            all_mask_img1=None, ## label semisupervised
                            lamb=1.0,
                            half_T_side_dense=10):
    #assert (not all_mask_img1.requires_grad)

    # bring x2 back into x1's spatial frame
    x2_outs_inv = perform_affine_tf(x2_outs, all_affine2_to_1)


    # zero out all irrelevant patches
    bn, k, h, w = x1_outs.shape
    #all_mask_img1 = all_mask_img1.view(bn, 1, h, w)  # mult, already float32
    #x1_outs = x1_outs * all_mask_img1  # broadcasts
    #x2_outs_inv = x2_outs_inv * all_mask_img1

    # sum over everything except classes, by convolving x1_outs with x2_outs_inv
    # which is symmetric, so doesn't matter which one is the filter
    x1_outs = x1_outs.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w
    x2_outs_inv = x2_outs_inv.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w

    # k, k, 2 * half_T_side_dense + 1,2 * half_T_side_dense + 1
    p_i_j = F.conv2d(x1_outs, weight=x2_outs_inv, padding=(half_T_side_dense,
                                                            half_T_side_dense))
    p_i_j = p_i_j.sum(dim=2, keepdim=False).sum(dim=2, keepdim=False)  # k, k

    # normalise, use sum, not bn * h * w * T_side * T_side, because we use a mask
    # also, some pixels did not have a completely unmasked box neighbourhood,
    # but it's fine - just less samples from that pixel
    current_norm = float(p_i_j.sum())
    p_i_j = p_i_j / current_norm

    # symmetrise
    p_i_j = (p_i_j + p_i_j.t()) / 2.

    # compute marginals
    p_i_mat = p_i_j.sum(dim=1).unsqueeze(1)  # k, 1
    p_j_mat = p_i_j.sum(dim=0).unsqueeze(0)  # 1, k

    # for log stability; tiny values cancelled out by mult with p_i_j anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_i_mat[(p_i_mat < EPS).data] = EPS
    p_j_mat[(p_j_mat < EPS).data] = EPS

    # maximise information
    loss = (-p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_i_mat) -
                        lamb * torch.log(p_j_mat))).sum()

    # for analysis only
    loss_no_lamb = (-p_i_j * (torch.log(p_i_j) - torch.log(p_i_mat) -
                                torch.log(p_j_mat))).sum()

    return loss, loss_no_lamb



class IIDSegmentationLoss(nn.Module):
    
    def __init__(self, half_T_side_dense = 10, lamb = 1):
        super(IIDSegmentationLoss, self).__init__()
        self.half_T_side_dense = half_T_side_dense
        self.lamb = lamb
        self.loss_fn = IID_segmentation_loss
    
    def forward(self, x1_outs, x2_outs, all_affine2_to_1):
        assert (x1_outs.requires_grad)
        assert (x2_outs.requires_grad)
        assert (not all_affine2_to_1.requires_grad)
        assert (x1_outs.shape == x2_outs.shape)

        
        return self.loss_fn(x1_outs, x2_outs, all_affine2_to_1,
                            lamb = self.lamb,
                            half_T_side_dense=self.half_T_side_dense)
        
 
class MaskedFocalLoss(nn.Module):
    """
    Focal Loss with class masking capability
    
    Args:
        gamma (float): Focusing parameter (>=0)
        alpha (Tensor): Class weighting (size: num_classes)
        ignore_class (int): Class index to exclude from loss
        reduction (str): 'mean', 'sum' or None
    
    Note:
        - alpha should contain weights for ALL classes (including ignored)
        - ignore_class is excluded BEFORE loss calculation
    """
    def __init__(self, gamma=2.0, alpha=None, ignore_class=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_class = ignore_class
        self.reduction = reduction

        # Validate alpha dimensions
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                raise TypeError("alpha should be a Tensor of size num_classes")

    def forward(self, logits, target):
        # Input shapes:
        # logits: (B, C, H, W) - raw model outputs
        # target: (B, H, W) - ground truth labels
        
        # Flatten tensors
        B, C, H, W = logits.size()
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        target_flat = target.reshape(-1)  # (B*H*W)

        # Create mask (exclude ignore_class)
        if self.ignore_class is not None:
            mask = target_flat != self.ignore_class
        else:
            mask = torch.ones_like(target_flat, dtype=torch.bool)

        # Apply mask
        logits_masked = logits_flat[mask]  # (N_valid, C)
        target_masked = target_flat[mask]  # (N_valid)

        # Handle empty masks (no valid pixels)
        if logits_masked.numel() == 0:
            return torch.tensor(0.0, device=logits.device)

        # Compute focal loss
        log_pt = F.log_softmax(logits_masked, dim=1)
        log_pt = log_pt.gather(1, target_masked.unsqueeze(1)).squeeze()
        pt = log_pt.exp()

        # Apply alpha if provided
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            at = alpha.gather(0, target_masked)
            log_pt = log_pt * at

        # Focal term
        loss = -1 * (1 - pt) ** self.gamma * log_pt

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction is None:
            return loss
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")
 
class FocalLoss(nn.Module):
    """
    Credits to  github.com/clcarwin/focal_loss_pytorch
    """
    def __init__(self, gamma=0, alpha=None, reduction=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction


    def forward(self, logits, target):
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # N,C,H,W => N,C,H*W
            logits = logits.transpose(1, 2)  # N,C,H*W => N,H*W,C
            logits = logits.contiguous().view(-1, logits.size(2))  # N,H*W,C => N*H*W,C
            target = target.reshape(-1, 1)
            
            
        logpt = F.log_softmax(logits, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.reduction is None:
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(
                "FocalLoss: reduction parameter not in list of acceptable values [\"mean\", \"sum\", None]")


class MaskedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, mean=True):
        """
        mean: return mean loss vs per element loss
        """
        super(MaskedCrossEntropyLoss, self).__init__()
        self.mean = mean
    
    def forward(self, logits, ground_truth):
        """
            Args:
                logits: (N,T,H,W,...,NumClasses)A Variable containing a FloatTensor of size
                    (batch, max_len, num_classes) which contains the
                    unnormalized probability for each class.
                target: A Variable containing a LongTensor of size
                    (batch, max_len) which contains the index of the true
                    class for each corresponding step.
                length: A Variable containing a LongTensor of size (batch,)
                    which contains the length of each data in a batch.
            Returns:
                loss: An average loss value masked by the length.
            """
        if type(ground_truth) == torch.Tensor:
            target = ground_truth
            mask = None
        elif len(ground_truth) == 1:
            target = ground_truth[0]
            mask = None
        elif len(ground_truth) == 2:
            target, mask = ground_truth
        else:
            raise ValueError("ground_truth parameter for MaskedCrossEntropyLoss is either (target, mask) or (target)")
        
        mask = target != 0
        if mask is not None:
            mask_flat = mask.reshape(-1, 1)  # (N*H*W x 1)
            nclasses = logits.shape[1] # (N, C, H, W)
            logits_flat = logits.reshape(-1, logits.size(1))  # (N*H*W x Nclasses)
            masked_logits_flat = logits_flat[mask_flat.repeat(1, nclasses)].view(-1, nclasses)
            target_flat = target.reshape(-1, 1)  # (N*H*W x 1)
            masked_target_flat = target_flat[mask_flat].unsqueeze(dim=-1).to(torch.int64)
        else:
            masked_logits_flat = logits.reshape(-1, logits.size(1))  # (N*H*W x Nclasses)
            masked_target_flat = target.reshape(-1, 1).to(torch.int64)  # (N*H*W x 1)
        masked_log_probs_flat = F.log_softmax(masked_logits_flat)  # (N*H*W x Nclasses)
        masked_losses_flat = -torch.gather(masked_log_probs_flat, dim=1, index=masked_target_flat)  # (N*H*W x 1)
        if self.mean:
            return masked_losses_flat.mean()
        return masked_losses_flat

# https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/dino_clstoken_loss.py#L12
class DINOLossV2(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, student_output_list, teacher_out_softmaxed_centered_list):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # TODO: Use cross_entropy_distribution here
        total_loss = 0
        for s in student_output_list:
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            for t in teacher_out_softmaxed_centered_list:
                loss = torch.sum(t[:lsm.shape[0]] * lsm, dim=-1)
                total_loss -= loss.mean()
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True
            
            
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        #dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) )

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
