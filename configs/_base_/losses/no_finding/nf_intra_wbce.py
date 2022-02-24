loss = dict(
    type='NIHNoFindingLoss',
    base_loss=dict(type='IntraWeightedBCEWithLogits', loss_weight=1.0),
)
