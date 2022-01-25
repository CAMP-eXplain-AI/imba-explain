from typing import Callable, Dict, Union

import torch
import torch.nn as nn
from ignite.engine import Engine
from torch.optim import Optimizer


def get_train_step_fn(classifier: nn.Module, criterion: nn.Module, optimizer: Optimizer,
                      device: Union[str, torch.device]) -> Callable:

    def _train_step_fn(engine: Engine, batch: Dict) -> float:
        classifier.train()
        imgs = batch['imgs'].to(device)
        targets = batch['targets'].to(device)
        preds = classifier(imgs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    return _train_step_fn


def get_eval_step_fn(classifier: nn.Module, device: Union[str, torch.device]):

    def _eval_step_fn(engine: Engine, batch: Dict) -> Dict:
        classifier.eval()
        with torch.no_grad():
            imgs = batch.pop('imgs')
            targets = batch.pop('targets')
            imgs = imgs.to(device)
            targets = targets.to(device)
            preds = classifier(imgs)

            batch.update({'preds': preds, 'targets': targets})
            return batch

    return _eval_step_fn
