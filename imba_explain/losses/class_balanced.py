from typing import Optional

import torch.nn as nn
from torch import Tensor

from .builder import LOSSES
from .cross_entropy import binary_cross_entropy


@LOSSES.register_module()
class ClassBalancedLoss(nn.Module):
    r""" Class-balanced loss. See `paper <https://arxiv.org/abs/1901.05555>`__ for details.

    Args:
        beta: the hyperparameter for computing the effective number of samples.

    """

    def __init__(self, beta: float, loss_weight: float = 1.0, reduction: str = 'mean') -> None:
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight
        self.reduction = reduction

        self.pos_weight = None

    def set_num_pos_neg(self, num_pos_neg: Tensor) -> None:
        # TODO to be implemented
        pass

    def forward(self,
                cls_scores: Tensor,
                label: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[float] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        loss_cls = self.loss_weight * binary_cross_entropy(
            cls_scores, label, weight=weight, reduction=reduction, avg_factor=avg_factor, pos_weight=self.pos_weight)
        return loss_cls
