import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name, num_classes, use_pretrained=False, width_mult=1.0, device='cpu'):
    """初始化选定模型"""
    weights_map = {
        'mobilenet_v2': models.MobileNet_V2_Weights.IMAGENET1K_V1,
        'resnet50': models.ResNet50_Weights.IMAGENET1K_V2,
        'efficientnet_b0': models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        'vit_b_16': models.ViT_B_16_Weights.IMAGENET21K_PLUS_V1
    }
    
    if model_name not in weights_map:
        raise ValueError(f"不支持的模型名称: {model_name}")

    # 预训练权重冲突检查：只有 width_mult 为 1.0 时才能使用官方权重
    if use_pretrained and width_mult != 1.0:
        print(f"警告: {model_name} 的预训练权重仅适用于 width_mult=1.0。")
        print(f"当前 width_mult={width_mult}，将忽略预训练权重并随机初始化。")
        use_pretrained = False

    weights = weights_map[model_name] if use_pretrained else None
    if weights:
        print(f"使用 {model_name} 的预训练权重")
    
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=weights, width_mult=width_mult)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes) # type: ignore
    elif model_name == 'resnet50':
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes) # type: ignore
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes) # type: ignore
    
    return model.to(device)
