""" Functions to create models """

import torch
import torch.nn as nn

def create(model_name: str, load_pretrain: bool, num_classes: int) -> nn.Module:
    if model_name == 'x3d_xs':
        return create_x3d_xs(load_pretrain, num_classes)
    elif model_name == 'x3d_xs_frozen':
        return create_x3d_xs_frozen(load_pretrain, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")

def create_x3d_xs(load_pretrain, num_classes):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=load_pretrain)
    model.blocks[5].proj = nn.Identity()
    return nn.Sequential(
        model,
        nn.Linear(2048, num_classes, bias=True),
    )

def create_x3d_xs_frozen(load_pretrain, num_classes):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=load_pretrain)
    model.blocks[5].proj = nn.Identity()
    for param in model.parameters():
        param.requires_grad = False
    return nn.Sequential(
        model,
        nn.Linear(2048, num_classes, bias=True),
    )

def load_model(model_name: str, load_pretrain: bool, num_classes: int, load_path: str) -> nn.Module:
    model = create(model_name, load_pretrain, num_classes)
    model.load_state_dict(torch.load(load_path))
    return model
    
