import os
import os.path as osp
from typing import Dict, List, Union

import cv2
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .builder import DATASETS, build_pipeline

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


@DATASETS.register_module()
class NIHClassificationDataset(Dataset):
    num_classes = 15

    def __init__(self, img_root: str, label_csv: str, pipeline: List[Dict]) -> None:
        super().__init__()

        self.img_root = img_root
        self.label_csv = label_csv

        self.img_files = os.listdir(self.img_root)
        label_df = pd.read_csv(self.label_csv)
        label_df.set_index('Image Index', inplace=True)
        # img_file to label. E.g., 0001_0000.png -> ['Edema', 'Pneumonia']
        self.img_to_label = {x: label_df.loc[x, 'Finding Labels'].split('|') for x in self.img_files}

        self.pipeline = build_pipeline(pipeline)

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, str]]:
        img_file = self.img_files[idx]
        img = cv2.cvtColor(cv2.imread(osp.join(self.img_root, img_file)), cv2.COLOR_BGR2RGB)
        img = self.pipeline(image=img)['image']
        result = {'img': img, 'img_file': img_file}

        label = self.img_to_label[img_file]
        label = torch.LongTensor([nih_cls_name_to_ind[disease_name] for disease_name in label])
        one_hot_label = torch.zeros(self.num_classes, dtype=torch.float32)
        one_hot_label.scatter_(0, label, 1.0)
        result.update({'target': one_hot_label})

        return result
