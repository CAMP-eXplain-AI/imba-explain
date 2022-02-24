_base_ = [
    '../../_base_/classifiers/truncated_densenet.py', '../../_base_/datasets/nih_binary_dataset.py',
    '../../_base_/losses/intra_wbce.py', '../../_base_/schedules/50e.py'
]

classifier = dict(num_classes=1)
