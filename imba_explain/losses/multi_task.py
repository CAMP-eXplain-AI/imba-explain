from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from .builder import LOSSES
from .cross_entropy import binary_cross_entropy


@LOSSES.register_module()
class MultiTaskLoss(nn.Module):
    r"""Multi-task loss. See `paper <https://arxiv.org/abs/1705.07115>`__ for details.

    Args:
        num_classes: Number of classes.
        loss_weight: weight factor multiplied with the multi-task loss.
        with_pos_weight: if True, the loss function will include learnable weights for positive answers of each class.
        reduction: The method to reduce the loss into a scalar.
        device: device to allocate the parameters.

    """

    def __init__(self,
                 num_classes: int = 15,
                 with_pos_weight: bool = False,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean',
                 device: Optional[Union[str, torch.device]] = None) -> None:
        super(MultiTaskLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.log_variance = nn.Parameter(torch.zeros(self.num_classes, device=device))

        if with_pos_weight:
            self.pos_weight = nn.Parameter(torch.ones(self.num_classes, device=device))
        else:
            self.pos_weight = None

    def forward(self,
                cls_scores: Tensor,
                label: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[float] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        # inter class weight, exp(- log_variance) = 1 / sigma^2. Please see the formulas in the original paper.
        class_weight = torch.exp(-self.log_variance)
        # intra class weight
        pos_weight = torch.relu(self.pos_weight) if self.pos_weight is not None else None
        loss_cls = binary_cross_entropy(
            cls_scores,
            label,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor,
            class_weight=class_weight,
            pos_weight=pos_weight)

        if reduction in ('mean', 'sum'):
            loss_cls += 0.5 * torch.sum(self.log_variance)

        loss_cls = self.loss_weight * loss_cls
        return loss_cls
