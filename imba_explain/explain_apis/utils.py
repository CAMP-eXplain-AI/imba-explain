import xml.etree.ElementTree as ET
from typing import Dict, Tuple, Union

import numpy as np


def load_bbox_annot(xml_file: str,
                    names_to_inds: Dict[str, int],
                    dtype: Union[str, np.dtype] = 'int') -> Tuple[np.ndarray, np.ndarray]:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    labels = []
    img_size = root.find('size')
    img_width = int(img_size.find('width').text)
    img_height = int(img_size.find('height').text)
    for obj in root.findall('object'):
        name = obj.find('name').text
        if names_to_inds is not None:
            label = names_to_inds[name]
        else:
            label = name

        bnd_box = obj.find('bndbox')
        bbox = [
            float(bnd_box.find('xmin').text),
            float(bnd_box.find('ymin').text),
            float(bnd_box.find('xmax').text),
            float(bnd_box.find('ymax').text)
        ]
        bboxes.append(bbox)
        labels.append(label)

    if len(bboxes) == 0:
        bboxes = np.zeros((0, 4), dtype=dtype)
        labels = np.zeros((0, ), dtype=int)
    else:
        bboxes = np.asarray(bboxes, dtype=dtype)
        bboxes = np.clip(
            bboxes, a_min=np.array([0, 0, 0, 0]), a_max=np.array([img_width, img_height, img_width, img_height]))
        labels = np.asarray(labels, dtype=int)

    return bboxes, labels
