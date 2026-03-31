import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='多模型图像分类训练脚本')
    
    # 必需参数
    parser.add_argument('--train_dir', type=str, default='./dataset/train', help='训练集目录路径')
    parser.add_argument('--val_dir', type=str, default='./dataset/val', help='验证集目录路径')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='mobilenet_v2', 
                        choices=['mobilenet_v2', 'resnet50', 'efficientnet_b0', 'vit_b_16'],
                        help='选择模型架构')
    parser.add_argument('--width_mult', type=float, default=1.0, help='MobileNetV2专用：网络宽度乘数(0.5-2.0)')
    parser.add_argument('--use_pretrained', action='store_true', help='使用预训练权重')
    
    # 训练参数
    parser.add_argument('--input_size', type=int, default=96, help='输入图像尺寸（推荐：MobileNet 224，ViT 384）')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default=None, help='自动选择设备(cuda/cpu)')
    
    # 优化参数
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减系数')
    
    args = parser.parse_args()
    
    # 自动设备选择
    args.device = torch.device(args.device if args.device else 
                              'cuda' if torch.cuda.is_available() else 'cpu')
    
    # 参数验证
    if args.model_name != 'mobilenet_v2' and args.width_mult != 1.0:
        print(f"警告：width_mult参数仅适用于MobileNetV2，已忽略该设置")
    if args.model_name == 'vit_b_16' and args.input_size < 224:
        args.input_size = 224
        print(f"ViT最低输入尺寸为224，已自动调整")
    
    return args
