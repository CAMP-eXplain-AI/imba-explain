import os
import os.path as osp
from typing import Dict, List, Union

import cv2
import pandas as pd
import torch
from ignite.utils import setup_logger
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
        # img_file to disease anmes. E.g., 0001_0000.png -> ['Edema', 'Pneumonia']
        self.img_to_diseases = {}
        self.num_pos_neg = torch.zeros((2, self.num_classes), dtype=torch.long)
        for img_file in self.img_files:
            diseases = label_df.loc[img_file, 'Finding Labels'].split('|')
            self.img_to_diseases.update({img_file: diseases})
            self.num_pos_neg[0] += self.one_hot_encode(diseases, dtype=torch.long)
        self.num_pos_neg[1] = len(self.img_files) - self.num_pos_neg[0]

        logger = setup_logger('imba-explain')
        log_nums = self.num_pos_neg.numpy()
        logger.info(f'Dataset under {self.img_root}: \nNumber of positive samples: {log_nums[0]};\n'
                    f'Number of negative samples {log_nums[1]}.')

        self.pipeline = build_pipeline(pipeline)

    def get_num_pos_neg(self) -> torch.Tensor:
        return self.num_pos_neg

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
