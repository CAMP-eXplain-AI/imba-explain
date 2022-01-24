import torch.nn as nn
from torch.optim import Optimizer
from typing import Union, Callable, Dict
from ignite.engine import Engine
import torch


def get_train_step_fn(classifier: nn.Module,
                      criterion: nn.Module,
                      optimizer: Optimizer, device: Union[str, torch.device]) -> Callable:
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