import logging
from abc import ABC
from typing import Dict, List, Optional

import numpy as np
import torch
from ignite.utils import setup_logger
from tabulate import tabulate
from torch.utils.data import Dataset

from .builder import build_pipeline


class ClassificationDataset(Dataset, ABC):

    def __init__(self, pipeline: List[Dict], clip_ratio: Optional[float] = None) -> None:
        super(ClassificationDataset, self).__init__()
        self.pipeline = build_pipeline(pipeline)
        self._imba_sampling_weights: Optional[np.ndarray] = None
        self._clip_ratio = clip_ratio

    def get_num_pos_neg(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        raise NotImplementedError

    @property
    def class_names(self) -> List[str]:
        raise NotImplementedError

    def imba_sampling_weights(self, logger: Optional[logging.Logger] = None) -> np.ndarray:
        if self._imba_sampling_weights is not None:
            return self._imba_sampling_weights
        else:
            num_pos_neg = self.get_num_pos_neg().numpy()
            # sampling weights when performing weighted sampling: 1 / num_pos
            _imba_sampling_weights = 1 / (num_pos_neg[0].astype(float) + 1e-6)
            if self._clip_ratio is not None:
                clip_val = np.quantile(_imba_sampling_weights, self._clip_ratio)
                _imba_sampling_weights = np.clip(_imba_sampling_weights, a_min=0, a_max=clip_val)
            _imba_sampling_weights /= _imba_sampling_weights.sum()
            self._imba_sampling_weights = _imba_sampling_weights

            logger = setup_logger('imba-explain') if logger is None else logger
            tabular_data = {'Name': self.class_names, 'Sampling Weights': self._imba_sampling_weights.round(3)}
            log_table = tabulate(tabular_data, headers='keys', tablefmt='pretty', numalign='left', stralign='left')
            log_str = 'Sampling weights of w.r.t. each class:\n'
            log_str += f'{log_table}'
            logger.info(log_str)

            return self._imba_sampling_weights

    def log_data_dist_info(self, class_names: List[str], logger: Optional[logging.Logger] = None) -> None:
        if logger is None:
            logger = setup_logger('imba-explain')
        num_pos_neg = self.get_num_pos_neg().numpy()
        tabular_data = {'Name': class_names, 'Positive Number': num_pos_neg[0], 'Negative Number': num_pos_neg[1]}
        log_table = tabulate(tabular_data, headers='keys', tablefmt='pretty', numalign='left', stralign='left')
        log_str = 'Numbers of positive/negative samples w.r.t. each class:\n'
        log_str += f'{log_table}'
        logger.info(log_str)

    def one_hot_encode(self,
                       names_to_inds: Dict,
                       diseases: List[str],
                       dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """

        Args:
            names_to_inds: Dict that maps class names to indices
            diseases: Disease names of an image sample.
            dtype: Data type of output Tensor.

        Returns:
            one-hot encoded label. Note that there can be multiple ones in the tensor, although the tensor is named
            "one-hot" label.
        """
        label = torch.LongTensor([names_to_inds[disease_name] for disease_name in diseases])
        one_hot_label = torch.zeros(self.num_classes, dtype=dtype)
        one_hot_label.scatter_(0, label, 1)
        return one_hot_label
