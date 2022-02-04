from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .builder import LOSSES
from .cross_entropy import binary_cross_entropy


@LOSSES.register_module()
class WeightedBCEWithLogits(nn.Module):

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 class_weight: Optional[Union[List, np.ndarray]] = None) -> None:
        super(WeightedBCEWithLogits, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

    @staticmethod
    def compute_pos_weight(label: torch.Tensor) -> torch.Tensor:
        """For each class, if the batch consists of all negative/positive samples, set the pos_weight to 1, otherwise
        num_neg / num_pos."""
        # one-hot encoded label: (num_samples, num_classes)
        num_samples = label.shape[0]
        num_pos = label.sum(0)
        num_neg: torch.Tensor = num_samples - num_pos

        # binary mask
        # 1 means set pos_weight of the class to num_neg / num_pos
        # 0 means set pos_weight of the class to 1
        mask = torch.logical_and(num_neg != 0, num_pos != 0).to(label)
        print(mask)
        pos_weight = num_neg / (num_pos + 1e-6)

        ones = torch.ones_like(pos_weight)
        pos_weight = pos_weight * mask + ones * (1 - mask)

        return pos_weight

    def forward(self,
                cls_score: Tensor,
                label: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[float] = None,
                reduction_override: Optional[str] = None,
                **kwargs: Any) -> torch.Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        pos_weight = self.compute_pos_weight(label)
        kwargs.update({'pos_weight': pos_weight})

        loss_cls = self.loss_weight * binary_cross_entropy(
            cls_score, label, weight, class_weight=class_weight, reduction=reduction, avg_factor=avg_factor, **kwargs)

        return loss_cls
