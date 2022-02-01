import os.path as osp
from copy import deepcopy
from typing import Union

import cv2
import numpy as np
import torch
from bbox_visualizer import add_multiple_labels, draw_multiple_rectangles
from captum.attr import LayerGradCam
from ignite.utils import manual_seed, setup_logger
from mmcv import Config
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..classifiers import build_classifier, get_module
from ..datasets import build_dataset


def show_grad_cam(cfg: Config,
                  ckpt: str,
                  plot_bboxes: bool = True,
                  device: Union[str, torch.device] = 'cuda:0',
                  with_pbar: bool = True) -> None:
    manual_seed(cfg.get('seed', 2022))
    logger = setup_logger('imba-explain')

    explain_set = build_dataset(cfg.data['explain'])
    data_loader_cfg = deepcopy(cfg.data['data_loader'])
    data_loader_cfg.update({'shuffle': False, 'drop_last': False})
    logger.info(f'Dataloader config: {data_loader_cfg}')
    explain_loader = DataLoader(explain_set, **data_loader_cfg)

    state_dict = torch.load(ckpt, map_location='cpu')
    logger.info(f'Using the checkpoint: {ckpt}')
    classifier = build_classifier(cfg.classifier)
    classifier.load_state_dict(state_dict)
    classifier.to(device)
    classifier.eval()

    target_layer = get_module(classifier, cfg.target_layer)
    grad_cam = LayerGradCam(classifier, target_layer)

    pbar = tqdm(total=len(explain_set)) if with_pbar else None

    for batch in explain_loader:
        imgs = batch['img']
        img_files = batch['img_file']
        bboxes_batch = batch['bboxes']
        labels_batch = batch['labels']

        for img, img_file, bboxes, labels in zip(imgs, img_files, bboxes_batch, labels_batch):
            img = img.to(device).unsqueeze(0)
            height, width = img.shape[2], img.shape[3]
            bboxes = bboxes.numpy().astype(int)
            labels = labels.numpy().astype(int)
            if labels.size == 0:
                continue

            unique_labels = np.unique(labels)
            for i, label in enumerate(unique_labels):
                attr_map = grad_cam.attribute(img, label, relu_attributions=True)
                if attr_map.shape[:2] != (1, 1):
                    raise ValueError(f'Attribution map has incorrect shape: {attr_map.shape}. '
                                     f'A valid shape should be (1, 1, height, width).')
                attr_map = attr_map.squeeze(0).squeeze(0).cpu().numpy()
                if attr_map.max() > 1 or attr_map.min() < 0:
                    raise ValueError(f'Attribution map has an invalid value range of '
                                     f'({attr_map.min()}, {attr_map.max()})')
                attr_map = (attr_map * 255).astype(np.uint8)
                attr_map = cv2.resize(attr_map, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
                # attr_map is in BGR mode.
                attr_map = cv2.applyColorMap(attr_map, cv2.COLORMAP_VIRIDIS)

                if plot_bboxes:
                    bboxes_to_draw = bboxes[labels == label]
                    labels_to_draw = labels[labels == label]
                    attr_map = draw_multiple_rectangles(attr_map, bboxes_to_draw, **cfg.bbox_cfg)
                    attr_map = add_multiple_labels(attr_map, labels_to_draw, bboxes_to_draw, **cfg.bbox_label_cfg)

                img_name = osp.splitext(osp.basename(img_file))[0]
                suffix = '' if i == 0 else f'_{i + 1}'
                out_path = osp.join(cfg.work_dir, f'{img_name}{suffix}.jpg')
                cv2.imwrite(out_path, attr_map)

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()
