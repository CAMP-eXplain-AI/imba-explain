import os.path as osp
from copy import deepcopy
from typing import Union

import cv2
import numpy as np
import torch
from captum.attr import LayerGradCam
from ignite.utils import manual_seed, setup_logger
from mmcv import Config
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..classifiers import build_classifier, get_module
from ..datasets import bbox_collate_fn, build_dataset
from .attr_normalizers import build_normalizer
from .bbox_visualizer import add_multiple_labels, draw_multiple_rectangles


def show_grad_cam(cfg: Config,
                  ckpt: str,
                  plot_bboxes: bool = True,
                  device: Union[str, torch.device] = 'cuda:0',
                  with_pbar: bool = True) -> None:
    manual_seed(cfg.get('seed', 2022))
    logger = setup_logger('imba-explain')

    explain_set = build_dataset(cfg.data['explain'])
    logger.info(f'Dataset under: {explain_set.img_root} contains {len(explain_set)} images')
    # a dict that maps label indices to bbox names
    inds_to_names = explain_set.get_inds_to_names()
    data_loader_cfg = deepcopy(cfg.data['data_loader'])
    data_loader_cfg.update({'shuffle': False, 'drop_last': False})
    logger.info(f'Dataloader config: {data_loader_cfg}')
    explain_loader = DataLoader(explain_set, collate_fn=bbox_collate_fn, **data_loader_cfg)

    state_dict = torch.load(ckpt, map_location='cpu')
    logger.info(f'Using the checkpoint: {ckpt}')
    classifier = build_classifier(cfg.classifier)
    classifier.load_state_dict(state_dict)
    classifier.to(device)
    classifier.eval()

    target_layer = get_module(classifier, cfg.target_layer)
    grad_cam = LayerGradCam(classifier, target_layer)
    # attribution map normalizer
    attr_normalizer = build_normalizer(cfg.attr_normalizer)

    pbar = tqdm(total=len(explain_set)) if with_pbar else None

    for batch in explain_loader:
        imgs = batch['img'].to(device)
        img_files = batch['img_file']
        bboxes_batch = batch['bboxes']
        labels_batch = batch['labels']

        with torch.no_grad():
            preds = classifier(imgs).sigmoid().cpu().numpy()

        for img, img_file, bboxes, labels, pred in zip(imgs, img_files, bboxes_batch, labels_batch, preds):
            img = img.unsqueeze(0)
            height, width = img.shape[2], img.shape[3]
            bboxes = bboxes.astype(int)
            labels = labels.astype(int)
            if labels.size == 0:
                continue

            unique_labels = np.unique(labels)
            for i, label in enumerate(unique_labels):
                attr_map = grad_cam.attribute(img, int(label), relu_attributions=True)
                if attr_map.shape[:2] != (1, 1):
                    raise ValueError(f'Attribution map has incorrect shape: {attr_map.shape}. '
                                     f'A valid shape should be (1, 1, height, width).')
                attr_map = attr_map.squeeze(0).squeeze(0).detach().cpu().numpy()
                # normalize the attribution map and convert it to an image
                attr_map = attr_normalizer(attr_map)
                attr_map = cv2.resize(attr_map, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
                # attr_map is in BGR mode.
                attr_map = cv2.applyColorMap(attr_map, cv2.COLORMAP_VIRIDIS)

                if plot_bboxes:
                    bboxes_to_draw = bboxes[labels == label]
                    labels_to_draw = labels[labels == label]
                    labels_to_draw = [f'{inds_to_names[i]}: {pred[i]:.2f}' for i in labels_to_draw]
                    attr_map = draw_multiple_rectangles(attr_map, bboxes_to_draw, **cfg.bbox_cfg)
                    attr_map = add_multiple_labels(attr_map, labels_to_draw, bboxes_to_draw, **cfg.bbox_label_cfg)

                img_name = osp.splitext(osp.basename(img_file))[0]
                suffix = '' if i == 0 else f'-{i + 1}'
                out_path = osp.join(cfg.work_dir, f'{img_name}{suffix}.jpg')
                cv2.imwrite(out_path, attr_map)

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()
