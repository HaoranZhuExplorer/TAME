import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

def prune_network(expert, prunning_ratio=0.95):
    parameters_to_prune = (
        (expert.net.features[0], 'weight'),
        (expert.net.features[1], 'weight'),
        (expert.net.features[4], 'weight'),
        (expert.net.features[5], 'weight'),
        (expert.net.features[8], 'weight'),
        (expert.net.features[9], 'weight'),
        (expert.net.features[11], 'weight'),
        (expert.net.features[12], 'weight'),
        (expert.net.features[15], 'weight'),
        (expert.net.features[16], 'weight'),
        (expert.net.features[18], 'weight'),
        (expert.net.features[19], 'weight'),
        (expert.net.features[22], 'weight'),
        (expert.net.features[23], 'weight'),
        (expert.net.features[25], 'weight'),
        (expert.net.features[26], 'weight'),
        (expert.net.classifier, 'weight'),
        (expert.net.features[0], 'bias'),
        (expert.net.features[1], 'bias'),
        (expert.net.features[4], 'bias'),
        (expert.net.features[5], 'bias'),
        (expert.net.features[8], 'bias'),
        (expert.net.features[9], 'bias'),
        (expert.net.features[11], 'bias'),
        (expert.net.features[12], 'bias'),
        (expert.net.features[15], 'bias'),
        (expert.net.features[16], 'bias'),
        (expert.net.features[18], 'bias'),
        (expert.net.features[19], 'bias'),
        (expert.net.features[22], 'bias'),
        (expert.net.features[23], 'bias'),
        (expert.net.features[25], 'bias'),
        (expert.net.features[26], 'bias'),
        (expert.net.classifier, 'bias')
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prunning_ratio,
    )
    
    