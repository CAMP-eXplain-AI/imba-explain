_base_ = [
    '../../_base_/classifiers/truncated_densenet.py', '../../_base_/datasets/nih_binary_dataset.py',
    '../../_base_/losses/no_finding/intra_wbce.py', '../../_base_/schedules/50e.py'
]
