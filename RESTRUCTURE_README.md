# MobileNet-Torch 模块说明

本项目已完成重构，采用模块化设计，方便移植和二次开发。

## 目录结构

- `models/`: 模型定义相关
  - `model_factory.py`: 模型工厂类，支持 MobileNetV2, ResNet50, EfficientNet, ViT 等。
- `data/`: 数据处理相关
  - `dataset.py`: 包含 `DiskCachedDataset`（硬盘缓存数据集）及 `DataLoader` 构建逻辑。
- `utils/`: 工具类
  - `args.py`: 命令行参数解析。
- `train.py`: 训练主入口脚本。

## 使用方法

### 训练模型
```bash
python train.py --model_name mobilenet_v2 --train_dir path/to/train --val_dir path/to/val --use_pretrained
```

### 主要参数
- `--model_name`: 选择模型 (mobilenet_v2, resnet50, efficientnet_b0, vit_b_16)
- `--batch_size`: 批大小
- `--input_size`: 输入图像尺寸
- `--use_pretrained`: 是否使用预训练权重
