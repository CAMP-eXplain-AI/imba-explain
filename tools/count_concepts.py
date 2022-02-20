from copy import deepcopy
from typing import Union

import mmcv
import torch
from argparse import ArgumentParser
from ignite.utils import setup_logger
from mmcv import DictAction
from torch.utils.data import DataLoader

from imba_explain.classifiers import build_classifier
from imba_explain.datasets import build_dataset
from imba_explain.explain_apis import ConceptDetector


def parse_args():
    parser = ArgumentParser('Count Number of Concepts.')
    parser.add_argument('config', help='Path to a configuration file.')
    parser.add_argument('ckpt', help='Path to a checkpoint file.')
    parser.add_argument('--with-pbar', action='store_true', help='Whether to use a progress bar.')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID.')

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair in xxx=yyy will be merged into config file.'
    )

    args = parser.parse_args()
    return args


def count_concepts(cfg: mmcv.Config,
                   ckpt: str,
                   with_pbar: bool = False,
                   device: Union[str, torch.device] = 'cuda:0') -> None:
    logger = setup_logger('imba-explain')

    explain_set = build_dataset(cfg.data['test'])
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

    concept_detector = ConceptDetector(cfg.img_size, quantile_threshold=cfg.quantile_threshold)

    concept_detector.set_classifier(classifier, target_layer=cfg.target_layer)
    concept_detector.detect(explain_loader, device=device, with_pbar=with_pbar)
    concept_detector.report(logger)


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    device = torch.device(f'cuda:{args.gpu_id}')
    count_concepts(cfg, ckpt=args.ckpt, with_pbar=args.with_pbar, device=device)


if __name__ == '__main__':
    main()
