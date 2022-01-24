import os.path as osp
from datetime import datetime
from typing import Union, Callable, Dict

import torch
import torch.nn as nn
from torch.optim import Optimizer
import mmcv
from mmcv import Config
from mmcv.runner import build_optimizer
from ..datasets import build_dataset
from ..classifiers import build_classifier
from ..losses import build_loss

from ignite.engine import Engine, Events
from ignite.utils import manual_seed, setup_logger
import ignite.distributed as idist


def train_classifier(local_rank: int, cfg: Config, work_dir) -> None:
    rank = idist.get_rank()
    manual_seed(cfg.get('seed', 2022) + rank)
    device = idist.device()

    logger = setup_logger('imba-explain',
                          filepath=osp.join(cfg.work_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    env_info_dict = mmcv.collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    logger.info(f'Config:\n{cfg.pretty_text}')

    train_set = build_dataset(cfg.data['train'])
    val_set = build_dataset(cfg.data['val'])
    test_set = build_dataset(cfg.data['test'])

    train_loader = idist.auto_dataloader(train_set, **cfg.data['data_loader'])
    val_loader = idist.auto_dataloader(val_set, **cfg.data['data_loader'])
    test_loader = idist.auto_dataloader(test_set, **cfg.data['data_loader'])
    epoch_length = len(train_loader)

    classifier = build_classifier(cfg.classifier)
    classifier.to(device)
    classifier = idist.auto_model(classifier, sync_bn=cfg.get('sync_bn', False))

    optimizer = build_optimizer(classifier, cfg.optimizier)
    criterion = build_loss(cfg.loss)
    criterion.to(device)
    trainer = Engine(get_train_step_fn(classifier, criterion, optimizer, device))
    trainer.logger = logger


    


    