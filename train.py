import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os
import hashlib
import multiprocessing
import argparse

class DiskCachedDataset(Dataset):
    """支持多模型预处理的硬盘缓存数据集"""
    def __init__(self, root, transform=None, cache_dir='./dataset_cache', model_name='mobilenet_v2'):
        self.source_dataset = datasets.ImageFolder(root)
        self.transform = transform
        self.classes = self.source_dataset.classes
        self.class_to_idx = self.source_dataset.class_to_idx
        
        # 生成包含模型名称的缓存目录
        root_path = os.path.abspath(root)
        hash_part = hashlib.md5((root_path + model_name).encode()).hexdigest()[:8]
        self.cache_dir = os.path.normpath(
            os.path.join(cache_dir, f"{model_name}_{os.path.basename(root)}_{hash_part}")
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.cache_paths = [
            os.path.join(self.cache_dir, 
                        f"{hashlib.md5(os.path.normpath(img_path).encode()).hexdigest()}.pt")
            for img_path, _ in self.source_dataset.imgs
        ]
        
        if not self._cache_exists():
            self._create_cache()

    def _cache_exists(self):
        return all(os.path.exists(p) for p in self.cache_paths)

    def _create_cache(self):
        for idx in tqdm(range(len(self.source_dataset)), 
                      desc=f'Creating cache {os.path.basename(self.cache_dir)}'):
            img, label = self.source_dataset[idx]
            if self.transform:
                img = self.transform(img)
            torch.save((img, label), self.cache_paths[idx])

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, idx):
        return torch.load(self.cache_paths[idx])

def parse_args():
    parser = argparse.ArgumentParser(description='多模型图像分类训练脚本')
    
    # 必需参数
    parser.add_argument('--train_dir', type=str, default='./dataset/train', help='训练集目录路径')
    parser.add_argument('--val_dir', type=str, default='./dataset/val', help='验证集目录路径')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='mobilenet_v2',choices=['mobilenet_v2', 'resnet50', 'efficientnet_b0', 'vit_b_16'],help='选择模型架构')
    parser.add_argument('--width_mult', type=float, default=1.0, help='MobileNetV2专用：网络宽度乘数(0.5-2.0)')
    parser.add_argument('--use_pretrained', action='store_true', help='使用预训练权重')
    
    # 训练参数
    parser.add_argument('--input_size', type=int, default=96, help='输入图像尺寸（推荐：MobileNet 224，ViT 384）')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default=None,help='自动选择设备(cuda/cpu)')
    
    # 优化参数
    parser.add_argument('--weight_decay', type=float, default=1e-4,help='权重衰减系数')
    
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

def get_model(args, num_classes):
    """初始化选定模型"""
    weights = None
    model = None
    
    # MobileNetV2
    if args.model_name == 'mobilenet_v2':
        if args.use_pretrained:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            print("使用ImageNet-1k预训练权重")
        model = models.mobilenet_v2(weights=weights, width_mult=args.width_mult)
        last_channel = max(1280, int(1280 * args.width_mult))
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )
    
    # ResNet50
    elif args.model_name == 'resnet50':
        if args.use_pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            print("使用ImageNet-1k V2预训练权重")
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # EfficientNet-B0
    elif args.model_name == 'efficientnet_b0':
        if args.use_pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            print("使用ImageNet-1k预训练权重")
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # Vision Transformer
    elif args.model_name == 'vit_b_16':
        if args.use_pretrained:
            weights = models.ViT_B_16_Weights.IMAGENET21K_PLUS_V1
            print("使用ImageNet-21k+预训练权重")
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    # 冻结预训练层（前5层）
    if args.use_pretrained:
        freeze_layers = 5
        for param in list(model.parameters())[:-freeze_layers]:
            param.requires_grad = False
    
    return model.to(args.device)

def get_transform(model_name, input_size):
    """获取模型专用预处理"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    
    # ViT需要调整插值方式
    if model_name == 'vit_b_16':
        return transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize
        ])
    
    # 其他模型使用默认
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize
    ])

def main(args):
    # 数据预处理
    transform = get_transform(args.model_name, args.input_size)
    
    # 加载数据集
    print(f"\n加载数据集（输入尺寸：{args.input_size}x{args.input_size}）")
    train_dataset = DiskCachedDataset(
        args.train_dir,
        transform=transform,
        cache_dir='./dataset_cache',
        model_name=args.model_name
    )
    val_dataset = DiskCachedDataset(
        args.val_dir,
        transform=transform,
        cache_dir='./dataset_cache',
        model_name=args.model_name
    )

    # 验证数据一致性
    assert train_dataset.classes == val_dataset.classes, "类别不一致！"
    num_classes = len(train_dataset.classes)
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True
    )

    # 初始化模型
    print(f"\n初始化{args.model_name.upper()}模型...")
    model = get_model(args, num_classes)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params/1e6:.2f}M")
    print(f"可训练参数: {trainable_params/1e6:.2f}M")

    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

    # 训练循环
    best_acc = 0.0
    print("\n开始训练...")
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        
        # 训练阶段
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
            inputs = inputs.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_dataset)

        # 验证阶段
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
        
        val_loss /= len(val_dataset)
        val_acc = correct / len(val_dataset)
        scheduler.step(val_acc)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'best_{args.model_name}.pth')
            print(f"保存最佳模型，准确率：{val_acc:.4f}")
        
        # 打印进度
        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e}")

    print(f"\n训练完成，最佳验证准确率：{best_acc:.4f}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    args = parse_args()
    main(args)