""" Functions to create models """

import torch
import torchvision
import torch.nn as nn

def create(model_name: str, load_pretrain: bool, num_classes: int) -> nn.Module:
    if model_name == 'x3d_xs':
        return create_x3d_xs(load_pretrain, num_classes)
    elif model_name == 'x3d_m':
        return create_x3d_m(load_pretrain, num_classes)
    elif model_name == 'lighter_x3d':
        return create_lighter_x3d(load_pretrain, num_classes)
    elif model_name == 'mobilenetv3_small':
        return create_mobilenetv3_small(load_pretrain, num_classes)
    elif model_name == 'mobilenetv3_small_DP':
        return create_mobilenetv3_small_DP(load_pretrain, num_classes)
    elif model_name == 'mobilenetv3_large':
        return create_mobilenetv3_large(load_pretrain, num_classes)
    elif model_name == 'mvitv2':
        return create_mvit_v2s(load_pretrain, num_classes)
    elif model_name == "swin_3d_s":
        return create_swin3d_s(load_pretrain, num_classes)
    elif model_name == "swin_3d_s_dp":
        return create_swin3d_s_dropout(load_pretrain, num_classes)
    elif model_name == "resnet152":
        return create_resnet152(load_pretrain, num_classes)
    elif model_name == "resnet50":
        return create_resnet50(load_pretrain, num_classes)
    elif model_name == "s3d":
        return create_s3d(load_pretrain, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")

def create_x3d_xs(load_pretrain, num_classes):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=load_pretrain)
    model.blocks[5].proj = nn.Identity()
    return nn.Sequential(
        model,
        nn.Linear(2048, num_classes, bias=True),
    )
    
def create_lighter_x3d(load_pretrain: bool, num_classes: int) -> nn.Module:
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=load_pretrain)

    def halve_conv_channels(layer):
        layer = layer.to("cuda")
        if isinstance(layer, torch.nn.Conv3d):
            new_out_channels = layer.out_channels // 2
            new_conv = torch.nn.Conv3d(
                in_channels=layer.in_channels,
                out_channels=new_out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=(layer.bias is not None)
            ).to("cuda")
            
            with torch.no_grad():
                new_conv.weight[:new_out_channels] = layer.weight[:new_out_channels].to("cuda")
                if layer.bias is not None:
                    new_conv.bias[:new_out_channels] = layer.bias[:new_out_channels].to("cuda")

            return new_conv
        else:
            return layer

    def update_model(model):
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Conv3d):
                setattr(model, name, halve_conv_channels(module))
            else:
                update_model(module)
                
    model.blocks[5].proj = nn.Identity()
    return nn.Sequential(
        model,
        nn.Linear(2048, num_classes, bias=True),
    )
    
def create_x3d_m(load_pretrain, num_classes):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=load_pretrain)
    model.blocks[5].proj = nn.Identity()
    return nn.Sequential(
        model,
        nn.Linear(2048, num_classes, bias=True),
    )
    
def create_mobilenetv3_small_DP(load_pretrain, num_classes):
    model = torchvision.models.mobilenet_v3_small(pretrained=load_pretrain)
    
    return nn.Sequential(
        model,
        nn.Linear(model.classifier[3].out_features, num_classes, bias=True),
    )

def create_mobilenetv3_small(load_pretrain, num_classes):
    model = torchvision.models.mobilenet_v3_small(pretrained=load_pretrain)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model

def create_mobilenetv3_large(load_pretrain, num_classes):
    model = torchvision.models.mobilenet_v3_large(pretrained=load_pretrain)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model

def create_mvit_v2s(load_pretrain, num_classes):
    model = torchvision.models.video.mvit_v2_s(pretrained=load_pretrain)
    model.head[1] = nn.Linear(model.head[1].in_features, num_classes)
    return model 

def create_s3d(load_pretrain, num_classes):
    model = torchvision.models.video.s3d(pretrained=load_pretrain)
    model.classifier[1] = nn.Conv3d(model.classifier[1].in_channels, 
        num_classes, 
        kernel_size=model.classifier[1].kernel_size, 
        stride=model.classifier[1].stride
    )
    return model

def create_swin3d_s(load_pretrain, num_classes):
    model = torchvision.models.video.swin3d_s(pretrained=load_pretrain)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

def create_swin3d_s_dropout(load_pretrain, num_classes):
    # Initialize the model
    model = torchvision.models.video.swin3d_t(pretrained=load_pretrain)
    
    # Define the dropout probability
    dropout_prob = 0.1
    
    # Modify all dropout layers in the model
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout_prob
    
    # Replace the final fully connected layer
    model.head = torch.nn.Sequential(
        torch.nn.Dropout(dropout_prob),
        torch.nn.Linear(model.head.in_features, num_classes)
    )
    print(f'Model looks like: {model}')
    return model

def create_resnet152(load_pretrain, num_classes):
    weights = torchvision.models.ResNet152_Weights.DEFAULT
    model = torchvision.models.resnet152(weights)
    last_fc = model.fc
    model.fc = nn.Linear(last_fc.in_features, num_classes)
    return model

def create_resnet50(load_pretrain, num_classes):
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights)
    last_fc = model.fc
    model.fc = nn.Linear(last_fc.in_features, num_classes)
    return model

def load_model(model_name: str, load_pretrain: bool, num_classes: int, load_path: str) -> nn.Module:
    model = create(model_name, load_pretrain, num_classes)
    model.load_state_dict(torch.load(load_path))
    return model