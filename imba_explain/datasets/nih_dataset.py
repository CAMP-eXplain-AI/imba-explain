import os
import os.path as osp
from typing import Dict, List, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from ignite.utils import setup_logger
from tabulate import tabulate
from torch import Tensor
from torch.utils.data import Dataset

from .builder import DATASETS, build_pipeline
from .utils import load_bbox_annot

nih_cls_name_to_ind = {
    'Atelectasis': 0,
    'Cardiomegaly': 1,
    'Consolidation': 2,
    'Edema': 3,
    'Effusion': 4,
    'Emphysema': 5,
    'Fibrosis': 6,
    'Hernia': 7,
    'Infiltration': 8,
    'Mass': 9,
    'No Finding': 10,
    'Nodule': 11,
    'Pleural_Thickening': 12,
    'Pneumonia': 13,
    'Pneumothorax': 14
}

# in xml files: infiltration is named as infiltrate
nih_bbox_name_to_ind = {
    'Atelectasis': 0,
    'Cardiomegaly': 1,
    'Effusion': 4,
    'Infiltrate': 8,
    'Mass': 9,
    'Nodule': 11,
    'Pneumonia': 13,
    'Pneumothorax': 14
}


@DATASETS.register_module()
class NIHClassificationDataset(Dataset):
    num_classes = 15

    def __init__(self,
                 img_root: str,
                 label_csv: str,
                 pipeline: List[Dict],
                 print_sampling_weights: bool = False) -> None:
        super().__init__()

        self.img_root = img_root
        self.label_csv = label_csv

        self.img_files = os.listdir(self.img_root)
        label_df = pd.read_csv(self.label_csv)
        label_df.set_index('Image Index', inplace=True)
        # img_file to disease names. E.g., 0001_0000.png -> ['Edema', 'Pneumonia']
        self.img_to_diseases = {}
        self.num_pos_neg = torch.zeros((2, self.num_classes), dtype=torch.long)
        for img_file in self.img_files:
            diseases = label_df.loc[img_file, 'Finding Labels'].split('|')
            self.img_to_diseases.update({img_file: diseases})
            self.num_pos_neg[0] += self.one_hot_encode(diseases, dtype=torch.long)
        self.num_pos_neg[1] = len(self.img_files) - self.num_pos_neg[0]

        logger = setup_logger('imba-explain')
        log_nums = self.num_pos_neg.numpy()
        # sampling weights when performing weighted sampling: 1 / num_pos
        _imba_sampling_weights = 1 / (log_nums[0].astype(float) + 1e-6)
        _imba_sampling_weights /= _imba_sampling_weights.sum()
        self._imba_sampling_weights = _imba_sampling_weights

        disease_names = [item[0] for item in sorted(nih_cls_name_to_ind.items(), key=lambda x: x[1])]
        tabular_data = {'Name': disease_names, 'Positive\nNumber': log_nums[0], 'Negative\nNumber': log_nums[1]}
        if print_sampling_weights:
            tabular_data.update({'Imbalanced\nSampling Weights': self._imba_sampling_weights})
        log_table = tabulate(tabular_data, headers='keys', tablefmt='github', floatfmt='.3f', numalign='left')
        log_str = f'Statistics of dataset under {self.img_root}\n'
        log_str += 'Numbers of positive/negative samples w.r.t. each class:\n'
        log_str += f'{log_table}'
        logger.info(log_str)

        self.pipeline = build_pipeline(pipeline)

    def get_num_pos_neg(self) -> torch.Tensor:
        return self.num_pos_neg

    @property
    def imba_sampling_weights(self) -> np.ndarray:
        return self._imba_sampling_weights

    def one_hot_encode(self, diseases: List[str], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """

        Args:
            diseases: Disease names of an image sample.
            dtype: Data type of output Tensor.

        Returns:
            one-hot encoded label. Note that there can be multiple ones in the tensor, although the tensor is named
            "one-hot" label.
        """
        label = torch.LongTensor([nih_cls_name_to_ind[disease_name] for disease_name in diseases])
        one_hot_label = torch.zeros(self.num_classes, dtype=dtype)
        one_hot_label.scatter_(0, label, 1)
        return one_hot_label

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, str]]:
        img_file = self.img_files[idx]
        img = cv2.cvtColor(cv2.imread(osp.join(self.img_root, img_file)), cv2.COLOR_BGR2RGB)
        img = self.pipeline(image=img)['image']
        result = {'img': img, 'img_file': img_file}

        diseases = self.img_to_diseases[img_file]
        one_hot_label = self.one_hot_encode(diseases)
        result.update({'target': one_hot_label})

        return result


@DATASETS.register_module()
class NIHDetectionDataset(Dataset):

    def __init__(self, img_root: str, annot_root: str, pipeline: List[Dict]) -> None:
        super(NIHDetectionDataset, self).__init__()
        self.img_root = img_root
        self.annot_root = annot_root

        default_args = {'bbox_params': A.BboxParams(format='pascal_voc', label_fields=['labels'])}
        self.pipeline = build_pipeline(pipeline, default_args=default_args)

        self.xml_files = os.listdir(self.annot_root)

        self.inds_to_names = {v: k for k, v in nih_bbox_name_to_ind.items()}

    def get_inds_to_names(self) -> Dict[int, str]:
        return self.inds_to_names

    def __len__(self) -> int:
        return len(self.xml_files)

    def __getitem__(self, idx: int) -> Dict:
        # base xml file name, e.g., xxxx_xxxx.xml
        xml_file = self.xml_files[idx]
        img_path = osp.join(self.img_root, osp.splitext(xml_file)[0] + '.png')
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        bboxes, labels = load_bbox_annot(osp.join(self.annot_root, xml_file), nih_bbox_name_to_ind, dtype='float')
        transformed = self.pipeline(image=img, bboxes=bboxes, labels=labels)
        img = transformed['image']
        bboxes = np.asarray(transformed['bboxes'], dtype=int)
        labels = np.asarray(transformed['labels'], dtype=int)
        return {'img_file': osp.basename(img_path), 'img': img, 'bboxes': bboxes, 'labels': labels}
