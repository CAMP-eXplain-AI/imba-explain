_base_ = ['../_base_/classifiers/resnet50.py', '../_base_/datasets/nih_class_dataset']

classifier = dict(num_classes=14)

# discard train and val datasets
data = dict(train=None, val=None, test=dict(select_classes=[0, 1, 2, 3, 4, 13, 14]))
pred_select_inds = [0, 1, 2, 3, 9, 11, 12]

class_metrics = dict(
    accuracy=dict(is_multilabel=True), roc_auc=dict(check_compute_fn=False), ap=dict(check_compute_fn=False))
