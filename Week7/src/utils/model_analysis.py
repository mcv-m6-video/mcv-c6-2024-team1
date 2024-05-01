""" 
This module contains functions to calculate the number of operations and parameters of a model. 
It uses the fvcore library to calculate the number of operations and parameters of a model.
Check out the fvcore library documentation for more information:
https://detectron2.readthedocs.io/en/latest/modules/fvcore.html
"""

import torch
from fvcore import nn


def calculate_operations(model, clip_length, crop_height, crop_width, channels: int = 3) -> int:
    """
    Calculate the number of operations of a model.

    Args:
        model (nn.Module): Model to calculate the number of operations.
        clip_length (int): Number of frames in a clip.
        crop_height (int): Height of the crop.
        crop_width (int): Width of the crop.

    Returns:
        int: Number of operations of the model.
    """
    mock_input = torch.randn(1, channels, clip_length, crop_height, crop_width).to("cuda")

    flops = nn.FlopCountAnalysis(model, mock_input)
    return flops.total()


def calculate_parameters(model) -> int:
    """
    Calculate the number of parameters of a model.

    Args:
        model (nn.Module): Model to calculate the number of parameters.

    Returns:
        int: Number of parameters of the model.
    """
    return nn.parameter_count(model)['']  # '' = global count


def calculate_operations_resnet(model, clip_length, crop_height, crop_width, channels: int = 3) -> int:
    """
    Calculate the number of operations of a model.

    Args:
        model (nn.Module): Model to calculate the number of operations.
        clip_length (int): Number of frames in a clip.
        crop_height (int): Height of the crop.
        crop_width (int): Width of the crop.

    Returns:
        int: Number of operations of the model.
    """
    mock_input = torch.randn(1, channels, crop_height, crop_width)

    flops = nn.FlopCountAnalysis(model, mock_input)
    return flops.total()