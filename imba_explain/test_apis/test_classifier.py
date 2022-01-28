import os.path as osp
from copy import deepcopy
from typing import Union

import ignite.distributed as idist
import mmcv
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC
from ignite.engine import Engine
from ignite.metrics import Accuracy
from ignite.utils import manual_seed, setup_logger
from mmcv import Config

from ..classifiers import build_classifier
from ..datasets import build_dataset
from ..train_apis import (MetricsTextLogger, PredictionsSaver, acc_metric_transform, get_eval_step_fn,
                          roc_auc_metric_transform)


def test_classifier(cfg: Config, ckpt: str, device: Union[str, torch.device] = 'cuda:0') -> None:
    """Test a classifier. This function only support the non-distributed setting.

    Args:
        cfg: config file.

    Returns:
        None
    """
    manual_seed(cfg.get('seed', 2022))

    logger = setup_logger('imba-explain')

    test_set = build_dataset(cfg.data['test'])
    data_loader_cfg = deepcopy(cfg.data['data_loader'])
    data_loader_cfg.update({'shuffle': False})
    test_loader = idist.auto_dataloader(test_set, **data_loader_cfg)

    state_dict = torch.load(ckpt, map_location='cpu')
    logger.info(f'Using the checkpoint: {ckpt}')
    classifier = build_classifier(cfg.classifier)
    classifier.load_state_dict(state_dict)
    classifier.to(device)
    classifier.eval()

    eval_step_fn = get_eval_step_fn(classifier, device)
    evaluator = Engine(eval_step_fn)
    evaluator.logger = logger

    pbar = ProgressBar(persist=True)
    pbar.attach(evaluator)

    test_metrics = {
        'accuracy': Accuracy(output_transform=acc_metric_transform, device=device, **cfg.class_metrics['accuracy']),
        'roc_auc': ROC_AUC(output_transform=roc_auc_metric_transform, device=device, **cfg.class_metrics['roc_auc']),
    }
    for name, metric in test_metrics.items():
        metric.attach(engine=evaluator, name=name)

    metrics_logger = MetricsTextLogger(logger=logger)
    metrics_logger.attach(evaluator)

    test_results_dir = osp.join(cfg.work_dir, 'test_results')
    mmcv.mkdir_or_exist(test_results_dir)
    preds_fp = osp.join(test_results_dir, 'predictions.csv')
    preds_saver = PredictionsSaver(preds_fp, logger=logger)
    preds_saver.attach(evaluator)

    evaluator.run(data=test_loader)
