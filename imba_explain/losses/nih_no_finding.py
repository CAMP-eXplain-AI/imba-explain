from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..datasets import NIHClassificationDataset
from .builder import LOSSES
from .utils import weight_reduce_loss


@LOSSES.register_module()
class NIHNoFindingLoss(nn.Module):
    """This is a lazy loss function wrapper.

    It is used to simulate the scenario of binary classification on the NIH dataset, where the target is to classify
    'Find pathologies' vs. 'Find nothing'. This is equivalent to assign 0 weights to other classes and 1 to the class
    'No Finding', and invert the class label of 'No Finding'.
    """

    def __init__(self, base_loss: Dict) -> None:
        super().__init__()
        self.base_loss = LOSSES.build(base_loss)
        # class index of 'No Finding'
        self.no_finding_ind = NIHClassificationDataset.name_to_ind['No Finding']

    def receive_data_dist_info(self, num_pos_neg: Tensor) -> None:
        if hasattr(self.base_loss, 'receive_data_dist_info'):
            self.base_loss.receive_data_dist_info(num_pos_neg)

    def forward(self,
                cls_score: Tensor,
                label: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[float] = None,
                reduction_override: Optional[bool] = None) -> Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.base_loss.reduction)

        # invert label of 'No Finding'
        label = torch.clone(label)
        label[:, self.no_finding_ind] = 1 - label[:, self.no_finding_ind]
        # assign 1.0 to class 'Find something' (i.e. inverted 'No Finding') and 0.0 to other classes
        find_something_weight = torch.zeros_like(cls_score)
        find_something_weight[:, self.no_finding_ind] = 1.0
        loss_cls = self.base_loss(cls_score, label, weight=weight, reduction_override='none')
        loss_cls = weight_reduce_loss(loss_cls, find_something_weight, reduction=reduction, avg_factor=avg_factor)
        return loss_cls
