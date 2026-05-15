import torch
import torch.nn as nn
from torchvision import models

def get_mobilenet_v2(num_classes=1000, pretrained=True):
    """
    获取 MobileNetV2 模型
    """
    if pretrained:
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
    else:
        model = models.mobilenet_v2(weights=None)
    
    # 修改最后一层分类器
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    return model

def get_mobilenet_v3_small(num_classes=1000, pretrained=True):
    """
    获取 MobileNetV3 Small 模型
    """
    if pretrained:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
    else:
        model = models.mobilenet_v3_small(weights=None)
        
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    
    return model

def get_mobilenet_v3_large(num_classes=1000, pretrained=True):
    """
    获取 MobileNetV3 Large 模型
    """
    if pretrained:
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
    else:
        model = models.mobilenet_v3_large(weights=None)
        
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    
    return model
