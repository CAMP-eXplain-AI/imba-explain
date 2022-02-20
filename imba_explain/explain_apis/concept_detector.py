import logging
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.utils import setup_logger
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm

from ..classifiers import get_module


class ConceptDetector:

    def __init__(self, img_size: Union[int, Tuple[int, int]] = 224, quantile_threshold: float = 0.99) -> None:
        self.classifier: Optional[nn.Module] = None
        self.hook_handle: Optional[RemovableHandle] = None

        self._feat_maps: List[torch.Tensor] = []
        self._num_channels: Optional[int] = None

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        elif isinstance(img_size, Sequence) and len(img_size) == 2:
            pass
        else:
            raise ValueError(f'Invalid img_size: {img_size}.')
        self.img_size: Tuple[int, int] = img_size
        self.quantile_threshold = quantile_threshold
        self.num_concepts = []

    def set_classifier(
        self,
        classifier: nn.Module,
        target_layer: str,
    ) -> None:
        self.classifier = classifier
        self.classifier.eval()
        layer = get_module(classifier, target_layer)

        def _forward_hook(module, _input, _output):
            self._feat_maps.append(_output.detach())
            return _output

        self.hook_handle = layer.register_forward_hook(_forward_hook)

    def clear_feat_maps(self) -> None:
        self._feat_maps.clear()

    def reset_num_concepts(self):
        self.num_concepts.clear()

    def remove_classifier_and_hook(self) -> None:
        self.classifier = None
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def report(self, logger: Optional[logging.Logger] = None) -> None:
        if self._num_channels is None:
            raise ValueError("The attribute '_num_channels' has not been set yet.")

        logger = setup_logger('imba-explain') if logger is None else logger
        num_samples = len(self.num_concepts)
        num_channels = self._num_channels
        min_val = np.min(self.num_concepts)
        max_val = np.max(self.num_concepts)
        avg_val = np.round(np.mean(self.num_concepts), 4)
        std_val = np.round(np.std(self.num_concepts), 4)
        median_val = np.median(self.num_concepts)

        tabular_data = {
            'Statistics': ['Samples', 'Channels', 'Min', 'Max', 'Average', 'Std', 'Median'],
            'Concepts': [num_samples, num_channels, min_val, max_val, avg_val, std_val, median_val]
        }

        table = tabulate(tabular_data, headers='keys', tablefmt='pretty', numalign='left', stralign='left')
        log_str = 'Number of Concepts:\n'
        log_str += f'{table}'
        logger.info(log_str)

    def detect(self,
               data_loader: DataLoader,
               device: Union[str, torch.device] = 'cuda:0',
               with_pbar: bool = False,
               remove_classifier_and_hook=True) -> None:
        pbar = tqdm(total=len(data_loader)) if with_pbar else None

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                img = batch['img'].to(device)
                _ = self.classifier(img)

                feat_map = self._feat_maps[0]
                num_samples, num_channels, feat_h, feat_w = feat_map.shape
                if i == 0:
                    self._num_channels = num_channels

                feat_map = F.interpolate(feat_map, size=self.img_size, mode='bilinear')
                # feat_map: (num_channels, num_samples, img_h, img_w)
                feat_map = torch.permute(feat_map, [1, 0, 2, 3])
                # feat_map: (num_channels, num_samples * img_h * img_w)
                feat_map = feat_map.reshape(num_channels, num_samples * self.img_size[0] * self.img_size[1])
                threshold = torch.quantile(feat_map, q=self.quantile_threshold, dim=1, keepdim=True)
                assert threshold.shape == (num_channels, 1)

                binary_mask = (feat_map >= threshold).astype(torch.float32)
                binary_mask = binary_mask.reshape(num_channels, num_samples, self.img_size[0], self.img_size[1])
                binary_mask = torch.permute(binary_mask, [1, 0, 2, 3])
                assert binary_mask.shape == (num_samples, num_channels) + self.img_size

                # binary_mask: (num_samples, num_channels, img_h, img_w)
                binary_mask = binary_mask.cpu().numpy()
                binary_mask = (binary_mask * 255.0).astype(np.uint8)
                for mask_single in binary_mask:
                    contours, _ = cv2.findContours(mask_single, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    self.num_concepts.append(len(contours))

                self.clear_feat_maps()

                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        if remove_classifier_and_hook:
            self.remove_classifier_and_hook()
