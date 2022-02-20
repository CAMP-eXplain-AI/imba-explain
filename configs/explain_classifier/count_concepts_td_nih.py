_base_ = [
    '../_base_/datasets/nih_class_dataset.py', '../_base_/explain/count_concepts.py',
    '../_base_/classifiers/truncated_densenet.py'
]

# Discard train and val datasets
data = dict(data_loader=dict(shuffle=False), train=None, val=None)

target_layer = 'features.norm5'
