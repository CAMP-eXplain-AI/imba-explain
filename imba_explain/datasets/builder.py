from typing import Dict, Optional

from mmcv import Registry

DATASETS = Registry('datasets')


def build_dataset(cfg: Dict, default_args: Optional[Dict] = None):
    return DATASETS.build(cfg, default_args=default_args)
