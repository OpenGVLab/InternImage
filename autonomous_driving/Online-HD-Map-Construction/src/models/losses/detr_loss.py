import torch
from torch import nn as nn
from torch.nn import functional as F
from mmdet.models.losses import l1_loss
from mmdet.models.losses.utils import weighted_loss
import mmcv

from mmdet.models.builder import LOSSES


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.
    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)

    return loss


@LOSSES.register_module()
class LinesLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, beta=0.5):
        """
            L1 loss. The same as the smooth L1 loss
            Args:
                reduction (str, optional): The method to reduce the loss.
                    Options are "none", "mean" and "sum".
                loss_weight (float, optional): The weight of loss.
        """

        super(LinesLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.beta = beta

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
                shape: [bs, ...]
            target (torch.Tensor): The learning target of the prediction.
                shape: [bs, ...]
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None. 
                it's useful when the predictions are not all valid.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss = smooth_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor, beta=self.beta)

        return loss*self.loss_weight


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def bce(pred, label, class_weight=None):
    """
        pred: B,nquery,npts
        label: B,nquery,npts
    """

    if label.numel() == 0:
        return pred.sum() * 0
    assert pred.size() == label.size()

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')

    return loss


@LOSSES.register_module()
class MasksLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MasksLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            xxx
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss = bce(pred, target, weight, reduction=reduction,
                   avg_factor=avg_factor)

        return loss*self.loss_weight

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def ce(pred, label, class_weight=None):
    """
        pred: B*nquery,npts
        label: B*nquery,
    """

    if label.numel() == 0:
        return pred.sum() * 0

    loss = F.cross_entropy(
        pred, label, weight=class_weight, reduction='none')

    return loss


@LOSSES.register_module()
class LenLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(LenLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            xxx
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss = ce(pred, target, weight, reduction=reduction,
                   avg_factor=avg_factor)

        return loss*self.loss_weight