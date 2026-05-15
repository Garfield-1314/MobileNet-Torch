import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
from PIL import Image

class CachedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.cache = [None] * len(subset)

    def __getitem__(self, index):
        if self.cache[index] is None:
            image, label = self.subset[index]
            self.cache[index] = (image, label)
        else:
            image, label = self.cache[index]
        
        # 应用实时数据增强（如随机翻转、归一化）
        if self.transform:
            image = self.transform(image)
            
        return image, label

    def __len__(self):
        return len(self.subset)

def get_dataloader(data_dir, batch_size=32, image_size=224):
    """
    获取训练和验证数据的 DataLoader
    启用了简单的内存缓存机制
    """
    
    # 定义基础转换（仅缩放和转为 Tensor，用于缓存）
    base_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # 定义实时增强转换（针对 Tensor）
    train_augment = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_augment = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载原始数据集（不带复杂增强，先转为基本 Tensor）
    raw_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), base_transform)
        for x in ['train', 'val']
    }

    # 使用自定义 Dataset 包装，实现 Lazy Cache
    # 注意：如果内存不足，请关闭此功能
    image_datasets = {
        'train': CachedDataset(raw_datasets['train'], transform=train_augment),
        'val': CachedDataset(raw_datasets['val'], transform=val_augment)
    }
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=1, pin_memory=True)
                  for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = raw_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names
