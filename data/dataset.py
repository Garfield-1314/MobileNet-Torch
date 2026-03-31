import os
import hashlib
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

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

def get_transform(model_name, input_size):
    """获取模型专用预处理"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    
    # ViT需要调整插值方式
    if model_name == 'vit_b_16':
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterationMode.BICUBIC),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # 其他模型使用默认
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize
        ])
    return transform

def get_dataloaders(args):
    """获取训练和验证数据加载器"""
    transform = get_transform(args.model_name, args.input_size)
    
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
    
    return train_loader, val_loader, len(train_dataset.classes)
