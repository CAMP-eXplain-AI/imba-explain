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

        # one-hot encoded label: (num_samples, num_classes)
        num_samples = label.shape[0]
        num_pos = label.sum(0)
        num_neg = num_samples - num_pos
        pos_weight = num_neg / num_pos
        kwargs.update({'pos_weight': pos_weight})

        loss_cls = self.loss_weight * binary_cross_entropy(
            cls_score, label, weight, class_weight=class_weight, reduction=reduction, avg_factor=avg_factor, **kwargs)

        return loss_cls
