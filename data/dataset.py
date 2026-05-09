import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_transform(model_name, input_size):
    """获取模型通用预处理"""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_dataloaders(args):
    """获取训练和验证数据加载器"""
    transform = get_transform(args.model_name, args.input_size)
    
    print(f"\n正在从以下路径加载数据集:")
    print(f"  训练集: {args.train_dir}")
    print(f"  验证集: {args.val_dir}")
    
    # 使用标准 ImageFolder，移除复杂的硬盘分块缓存以保持简单性
    train_dataset = datasets.ImageFolder(args.train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(args.val_dir, transform=transform)

    # 验证数据一致性
    if train_dataset.classes != val_dataset.classes:
        print(f"警告: 训练集类别 ({len(train_dataset.classes)}) 与验证集类别 ({len(val_dataset.classes)}) 不一致！")
    
    # 自动计算合理的 num_workers
    num_workers = min(os.cpu_count(), 8) if os.name == 'nt' else os.cpu_count()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, len(train_dataset.classes)
