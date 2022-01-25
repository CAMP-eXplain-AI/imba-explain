from typing import Dict, Tuple

from torch import Tensor


def metrics_transform(batch: Dict) -> Tuple[Tensor, Tensor]:
    return batch['preds'], batch['targets']
