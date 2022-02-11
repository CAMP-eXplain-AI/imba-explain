from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np

from .builder import NORMALIZERS


class BaseNormalizer(metaclass=ABCMeta):

    @abstractmethod
    def normalize(self, attr_map: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        """Normalize the attribution map.

        Args:
            attr_map: attribution maps with shape (height, width).
            args: Other arguments.
            kwargs: Other keyword arguments.

        Returns:
            normalized attribution maps.
        """
        pass

    def to_img(self, attr_map: np.ndarray) -> np.ndarray:
        attr_map = np.clip(attr_map, a_min=0, a_max=1)
        attr_map = (attr_map * 255).astype(np.uint8)
        return attr_map

    def __call__(self, attr_map: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        """First normalize the attribution map, and then convert it to a single-channel 8-bit image.

        Args:
            attr_map: Attribution map with shape (height, width).
            args: Other arguments of the `normalize` function.
            kwargs: Other keyword arguments of the `normalize` function.

        Returns:
            The attribution map as a single-channel 8-bit image.
        """
        attr_map = self.normalize(attr_map)
        return self.to_img(attr_map)


@NORMALIZERS.register_module()
class IdentityNormalizer(BaseNormalizer):

    def __init__(self) -> None:
        super(IdentityNormalizer, self).__init__()

    def normalize(self, attr_map: np.ndarray) -> np.ndarray:
        return attr_map


@NORMALIZERS.register_module()
class ScaleNormalizer(BaseNormalizer):

    def __init__(self, scale: float = 1.0) -> None:
        super(ScaleNormalizer, self).__init__()
        self.scale = scale

    def normalize(self, attr_map: np.ndarray, *args, **kwargs) -> np.ndarray:
        return self.scale * attr_map


@NORMALIZERS.register_module()
class MinMaxNormalizer(BaseNormalizer):

    def __init__(self) -> None:
        super(MinMaxNormalizer, self).__init__()

    def normalize(self, attr_map: np.ndarray, *args, **kwargs) -> np.ndarray:
        min_val = np.min(attr_map)
        max_val = np.max(attr_map)
        return (attr_map - min_val) / (max_val - min_val + 1e-8)
