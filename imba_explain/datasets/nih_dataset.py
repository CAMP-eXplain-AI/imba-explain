import os
import os.path as osp
from typing import Dict, List, Optional, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .builder import DATASETS, build_pipeline
from .cls_dataset import ClassificationDataset
from .utils import load_bbox_annot


@DATASETS.register_module()
class NIHClassificationDataset(ClassificationDataset):
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

    def __init__(self, img_root: str, label_csv: str, pipeline: List[Dict], clip_ratio: Optional[float] = None) -> None:
        super().__init__(pipeline=pipeline, clip_ratio=clip_ratio)

        self.img_root = img_root
        self.label_csv = label_csv

        self.img_files = os.listdir(self.img_root)
        label_df = pd.read_csv(self.label_csv)
        label_df.set_index('Image Index', inplace=True)

        # img_file to disease names. E.g., 0001_0000.png -> ['Edema', 'Pneumonia']
        self.img_to_diseases = {}

        self._class_names = [item[0] for item in sorted(self.nih_cls_name_to_ind.items(), key=lambda x: x[1])]
        self._num_classes = len(self._class_names)
        self.num_pos_neg = torch.zeros((2, self.num_classes), dtype=torch.long)

        for img_file in self.img_files:
            diseases = label_df.loc[img_file, 'Finding Labels'].split('|')
            self.img_to_diseases.update({img_file: diseases})
            self.num_pos_neg[0] += self.one_hot_encode(self.nih_cls_name_to_ind, diseases, dtype=torch.long)
        self.num_pos_neg[1] = len(self.img_files) - self.num_pos_neg[0]

        self.log_data_dist_info(class_names=self._class_names)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def get_num_pos_neg(self) -> torch.Tensor:
        return self.num_pos_neg

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, str]]:
        img_file = self.img_files[idx]
        img = cv2.cvtColor(cv2.imread(osp.join(self.img_root, img_file)), cv2.COLOR_BGR2RGB)
        img = self.pipeline(image=img)['image']
        result = {'img': img, 'img_file': img_file}

        diseases = self.img_to_diseases[img_file]
        one_hot_label = self.one_hot_encode(self.nih_cls_name_to_ind, diseases)
        result.update({'target': one_hot_label})

        return result


@DATASETS.register_module()
class NIHDetectionDataset(Dataset):
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

    def __init__(self, img_root: str, annot_root: str, pipeline: List[Dict]) -> None:
        super(NIHDetectionDataset, self).__init__()
        self.img_root = img_root
        self.annot_root = annot_root

        default_args = {'bbox_params': A.BboxParams(format='pascal_voc', label_fields=['labels'])}
        self.pipeline = build_pipeline(pipeline, default_args=default_args)

        self.xml_files = os.listdir(self.annot_root)

        self.inds_to_names = {v: k for k, v in self.nih_bbox_name_to_ind.items()}

    def get_inds_to_names(self) -> Dict[int, str]:
        return self.inds_to_names

    def __len__(self) -> int:
        return len(self.xml_files)

    def __getitem__(self, idx: int) -> Dict:
        # base xml file name, e.g., xxxx_xxxx.xml
        xml_file = self.xml_files[idx]
        img_path = osp.join(self.img_root, osp.splitext(xml_file)[0] + '.png')
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        bboxes, labels = load_bbox_annot(osp.join(self.annot_root, xml_file), self.nih_bbox_name_to_ind, dtype='float')
        transformed = self.pipeline(image=img, bboxes=bboxes, labels=labels)
        img = transformed['image']
        bboxes = np.asarray(transformed['bboxes'], dtype=int)
        labels = np.asarray(transformed['labels'], dtype=int)
        return {'img_file': osp.basename(img_path), 'img': img, 'bboxes': bboxes, 'labels': labels}
