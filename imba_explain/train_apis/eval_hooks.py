from collections import defaultdict
from itertools import chain
from logging import Logger
from typing import Optional

import ignite.distributed as idist
import numpy as np
import pandas as pd
import torch
from ignite.engine import Engine, Events
from ignite.utils import setup_logger


class MetricsTextLogger:

    def __init__(self, logger: Optional[Logger] = None) -> None:
        self.logger = logger if logger is not None else setup_logger('imba-explain')

    def _log_metrics(self, evaluator: Engine, trainer: Optional[Engine] = None) -> None:
        if trainer is not None:
            epoch = trainer.state.epoch
            max_epochs = trainer.state.max_epochs
            log_str = f'Epoch [{epoch}/{max_epochs}]: '
        else:
            log_str = 'Evaluation metrics: '
        log_str += '; '.join([f'{name}: {val:.4f}' for name, val in evaluator.state.metrics])
        self.logger.info(log_str)

    def attach(self, evaluator: Engine, trainer: Optional[Engine] = None) -> None:
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, self._log_metrics, trainer)


class PredictionsSaver:

    def __init__(self, file_path: str, logger: Optional[Logger] = None) -> None:
        if idist.get_world_size() > 1:
            raise RuntimeError(f'PredictionSaver only support the non-distributed setting, '
                               f'but the world size is {idist.get_world_size()}.')
        if file_path.endswith('.csv'):
            raise ValueError(f"file path must be a string ending with '.csv', but got {file_path}.")

        self.file_path = file_path
        self.logger = logger if logger is not None else setup_logger('imba-explain')
        self.buffers = defaultdict(list)
        self._has_saved = False

    def _init_buffers(self, engine: Engine) -> None:
        self.buffers.clear()

    def _update_buffers(self, engine: Engine) -> None:
        output = engine.state.output
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            self.buffers[key].append(value)

    def _save_buffers(self, engine: Engine) -> None:
        if not self._has_saved:
            for key, value in self.buffers.items():
                if isinstance(value, np.ndarray):
                    value = np.concatenate(value, 0)
                elif isinstance(value, (list, tuple)):
                    value = list(chain.from_iterable(value))
                else:
                    raise TypeError(f'Invalid type of data stored in the buffer: {value.__class__.__name__}.')
                self.buffers.update({key: value})

        df = pd.DataFrame.from_dict(self.buffers)
        df.to_csv(self.file_path)
        self.logger.info(f'Predictions have been saved to {self.file_path}.')
        self._has_saved = True

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.EPOCH_STARTED, self._init_buffers)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._update_buffers)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._save_buffers)
