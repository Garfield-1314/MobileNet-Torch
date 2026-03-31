import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name, num_classes, use_pretrained=False, width_mult=1.0, device='cpu'):
    """初始化选定模型"""
    weights = None
    model = None
    
    # MobileNetV2
    if model_name == 'mobilenet_v2':
        if use_pretrained:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            print("使用ImageNet-1k预训练权重")
        model = models.mobilenet_v2(weights=weights, width_mult=width_mult)
        last_channel = max(1280, int(1280 * width_mult))
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )
    
    # ResNet50
    elif model_name == 'resnet50':
        if use_pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            print("使用ImageNet-1k V2预训练权重")
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # EfficientNet-B0
    elif model_name == 'efficientnet_b0':
        if use_pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            print("使用ImageNet-1k预训练权重")
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # Vision Transformer
    elif model_name == 'vit_b_16':
        if use_pretrained:
            weights = models.ViT_B_16_Weights.IMAGENET21K_PLUS_V1
            print("使用ImageNet-21k+预训练权重")
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"不支持的模型名称: {model_name}")
    
    # 冻结预训练层（前5层）
    if use_pretrained:
        freeze_layers = 5
        for param in list(model.parameters())[:-freeze_layers]:
            param.requires_grad = False
    
    return model.to(device)
