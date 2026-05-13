import torch
from .metrics.numpy_metrics import get_classification_metrics


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

def build_backbone_from_cfg(cfg):
    from models.TSViT.swinTSViT import DINOTSViT
    model = None
    if	cfg.model_name == 'DinoTSViT':
        model = DINOTSViT(cfg)
        
    return model