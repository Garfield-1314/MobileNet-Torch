# MobileNet-Torch

本项目是一个基于 PyTorch 的深度学习工具链，涵盖了从模型训练、转换到移动端部署（TFLite）的完整流程。项目特别针对 MobileNetV2 进行了优化，并提供了一套自建的 ONNX 到 TFLite 转换工具。

## 🌟 核心功能

- **高效训练**: 支持 MobileNetV2、ResNet50、EfficientNet 和 ViT 等多种模型。
- **硬盘缓存**: 引入 `DiskCachedDataset`，通过预处理缓存加速数据读取。
- **多格式支持**: 支持将 PyTorch (`.pth`) 模型导出为 ONNX、TensorFlow SavedModel 及 TFLite 格式。
- **精准转换**: 内置 `onnx2tflite` 工具，通过 Keras 算子重构实现高精度的模型转换（精度误差 < 1e-5）。

## 🛠️ 环境要求

- **Python**: 3.11
- **CUDA**: 11.8
- **cuDNN**: 8.9.7

### 安装依赖

```bash
# 安装 PyTorch 环境（conda）
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# 其他常用依赖
pip install onnx onnx-tf onnxsim tensorflow keras
```

## 📂 目录结构

```text
├── models/                 # [NEW] 模型定义模块 (MobileNet, ResNet, EfficientNet, ViT)
├── data/                   # [NEW] 数据处理模块 (DiskCachedDataset, Dataloader)
├── utils/                  # [NEW] 通用工具模块 (参数解析, 日志管理)
├── train.py                # [REFACTORED] 模型训练主入口 (支持多模型配置)
├── export_to_onnx.py       # PyTorch (.pth) 转 ONNX 脚本
├── onnx_2_tf.py            # ONNX 转 TensorFlow SavedModel
├── gpu_test.py             # CUDA/cuDNN 环境检测工具
├── onnx2tflite/            # 核心转换工具库 (ONNX -> TFLite/H5)
│   ├── onnx2tflite/        # 转换器实现 (按层解析与重构)
│   └── test/               # 转换器功能测试脚本
└── README.md
```

## 🚀 快速开始

### 1. 模型训练
将数据集放置在 `./dataset/train` 和 `./dataset/val` 目录下，然后运行：
```bash
# 基本训练
python train.py

# 指定模型与参数
python train.py --model_name mobilenet_v2 --batch_size 64 --input_size 224 --use_pretrained
```
*提示：项目采用模块化设计，支持通过命令行参数灵活切换模型架构（如 `resnet50`, `efficientnet_b0`, `vit_b_16`）。*

### 2. 导出为 ONNX
训练完成后，将生成的 `best_model.pth` 转换为 ONNX 格式：
```bash
python export_to_onnx.py
```

### 3. 转换为 TFLite
使用内置工具将 ONNX 模型转换为移动端格式：
```bash
# 使用 onnx2tflite 转换工具
python -m onnx2tflite --weights mobilenetv2_custom.onnx
```

## 📊 转换精度
本项目提供的 `onnx2tflite` 在转换过程中会进行输出校验，确保转换前后的推理结果一致性。

