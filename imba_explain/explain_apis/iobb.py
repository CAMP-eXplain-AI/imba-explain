from typing import Optional, Sequence

import numpy as np

from .pointing_game import bboxes_to_mask


class IoBB:

    def __init__(self, attr_threshold: Optional[float] = None) -> None:
        if attr_threshold is not None and attr_threshold >= 1.0 or attr_threshold <= 0.0:
            raise ValueError(f"'attr_threshold' should be in range (0, 1), but got {attr_threshold}")
        self.attr_threshold = attr_threshold

    def evaluate(self, attr_map: np.ndarray, bboxes: Sequence[np.ndarray]) -> float:
        if len(attr_map.shape) != 2:
            raise ValueError(f'attribution map should be in shape (height, width), but got {attr_map.shape}')
        if np.max(attr_map) > 1.0 or np.min(attr_map) < 0.0:
            raise ValueError(f'attribution map should have value range [0, 1], '
                             f'but got [{np.min(attr_map):.4f}, {np.max(attr_map):.4f}]')

        gt_mask = bboxes_to_mask(bboxes, shape=attr_map.shape, dtype=float)
        if self.attr_threshold is not None:
            attr_map = (attr_map >= self.attr_threshold).astype(gt_mask.dtype)

        inter = (gt_mask * attr_map).sum()
        bb = gt_mask.sum()
        iobb = inter / (bb + 1e-8)
        return iobb
