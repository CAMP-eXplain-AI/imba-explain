from typing import Dict, Tuple, Union

from torch import Tensor


def prob_transform(batch: Dict[str, Union[Tensor, str]]) -> Tuple[Tensor, Tensor]:
    pred = batch['pred'].sigmoid()
    target = batch['target']
    return (pred >= 0.5).to(target), target


def logits_transform(batch: Dict[str, Union[Tensor, str]]) -> Tuple[Tensor, Tensor]:
    return batch['pred'], batch['target']
