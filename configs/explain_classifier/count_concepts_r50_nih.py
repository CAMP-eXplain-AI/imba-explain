_base_ = [
    '../_base_/datasets/nih_class_dataset.py', '../_base_/explain/count_concepts.py',
    '../_base_/classifiers/resnet50.py'
]

# Discard train and val datasets
data = dict(data_loader=dict(shuffle=False), train=None, val=None)
