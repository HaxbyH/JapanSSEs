"""Script with helper function."""
from lrp_methods.lrp_layers import *
# from lrp_layers import *

import torch

def layers_lookup() -> dict:
    """Lookup table to map network layer to associated LRP operation.

    Returns:
        Dictionary holding class mappings.
    """
    lookup_table = {
        torch.nn.modules.linear.Linear: RelevancePropagationLinear,
        torch.nn.modules.conv.Conv2d: RelevancePropagationConv2d,
        torch.nn.modules.conv.Conv1d: RelevancePropagationConv1d,
        torch.nn.modules.activation.ReLU: RelevancePropagationReLU,
        torch.nn.modules.dropout.Dropout: RelevancePropagationDropout,
        torch.nn.modules.flatten.Flatten: RelevancePropagationFlatten,
        torch.nn.modules.pooling.AvgPool2d: RelevancePropagationAvgPool2d,
        torch.nn.modules.pooling.AvgPool1d: RelevancePropagationAvgPool1d,
        torch.nn.modules.pooling.MaxPool2d: RelevancePropagationMaxPool2d,
        torch.nn.modules.pooling.AdaptiveAvgPool2d: RelevancePropagationAdaptiveAvgPool2d,
        torch.nn.modules.Sigmoid: RelevancePropagationIdentity,
        torch.nn.modules.BatchNorm2d: RelevancePropagationIdentity,
        torch.nn.modules.BatchNorm1d: RelevancePropagationIdentity,
        torch.nn.ZeroPad2d: RelevanceZeroPad,
        
    }
    return lookup_table
