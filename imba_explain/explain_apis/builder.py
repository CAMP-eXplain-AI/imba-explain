from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import torch.nn as nn
from mmcv import Registry
from torch import Tensor

NORMALIZERS = Registry('Attribution Normalizers')
ATTRIBUTIONS = Registry('Attribution Methods')


def build_normalizer(cfg: Dict, default_args: Optional[Dict] = None) -> Any:
    return NORMALIZERS.build(cfg, default_args=default_args)


def build_attributions(cfg: Dict, default_args: Optional[Dict] = None) -> Any:
    return ATTRIBUTIONS.build(cfg, default_args=default_args)


class BaseAttribution(metaclass=ABCMeta):

    def __init__(self, attr_normalizer: Dict) -> None:
        self.attr_normalizer = build_normalizer(attr_normalizer)

    @abstractmethod
    def set_classifier(self, classifier: nn.Module) -> None:
        pass

    @abstractmethod
    def attribute(self, img: Tensor, label: Union[int, Tensor, np.ndarray], *args: Any, **kwargs: Any) -> np.ndarray:
        pass

    def attribute_and_normalize(self, img: Tensor, label: Union[int, Tensor, np.ndarray], *args: Any,
                                **kwargs: Any) -> np.ndarray:
        attr_map = self.attribute(img, label, *args, **kwargs)
        # normalize the attribution map and convert it to an image
        return self.attr_normalizer(attr_map)

    def __call__(self, img: Tensor, label: Union[int, Tensor, np.ndarray], *args: Any, **kwargs: Any) -> np.ndarray:
        return self.attribute_and_normalize(img, label, *args, **kwargs)
