import os
import torch
import torch.nn as nn
from models.model_factory import get_model
from utils.args import parse_args

def main():
    args = parse_args()
    
    # 1. 使用工厂模式加载模型
    # 注意：导出通常不需要预训练权重（因为会加载本地 .pth）
    print(f"正在初始化 {args.model_name} 模型...")
    
    # 模拟从 dataloader 获取类别数，如果没有指定则尝试根据 checkpoint 推断或使用默认
    num_classes = args.num_classes if hasattr(args, 'num_classes') else 15
    
    model = get_model(
        args.model_name, 
        num_classes, 
        width_mult=args.width_mult, 
        device='cpu'
    )

    # 2. 加载 .pth 文件
    model_path = f"best_{args.model_name}.pth"
    if not os.path.exists(model_path):
        # 兼容旧的文件名
        if os.path.exists("best_model.pth"):
            model_path = "best_model.pth"
        else:
            print(f"错误: 找不到模型文件 {model_path} 或 best_model.pth")
            return

    print(f"正在从 {model_path} 加载权重...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    # 3. 导出 ONNX
    dummy_input = torch.randn(1, 3, args.input_size, args.input_size)
    onnx_path = f"{args.model_name}_optimized.onnx"
    
    print(f"正在导出模型到 {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11
    )

    print("导出成功！")

if __name__ == '__main__':
    main()
