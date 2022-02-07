_base_ = [
    '../_base_/datasets/nih_detect_dataset.py', '../_base_/explain/grad_cam.py',
    '../_base_/classifiers/truncated_densenet.py'
]

target_layer = 'features.norm5'
attr_normalizer = dict(type='MinMaxNormalizer')
