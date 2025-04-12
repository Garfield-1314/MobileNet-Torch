import torch

def check_gpu_availability():
    # 打印PyTorch版本信息
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检测CUDA可用性
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {'是' if cuda_available else '否'}")
    
    if cuda_available:
        # 显示CUDA版本和驱动信息
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU驱动信息: {torch.cuda.get_device_name()}")
        
        # 显示GPU设备数量
        device_count = torch.cuda.device_count()
        print(f"检测到GPU数量: {device_count}")
        
        # 遍历所有GPU并显示详细信息
        for i in range(device_count):
            print(f"\nGPU {i} 详细信息:")
            print(f"  设备名称: {torch.cuda.get_device_name(i)}")
            print(f"  计算能力: {torch.cuda.get_device_capability(i)}")
            print(f"  显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # 设置默认设备并测试张量计算
        device = torch.device('cuda:0' if cuda_available else 'cpu')
        print(f"\n当前默认设备: {device}")
        
        # 创建测试张量并验证设备
        x = torch.randn(3, 3).to(device)
        print(f"测试张量设备: {x.device}")
        print("GPU计算测试成功!")
    else:
        print("\n注意: 将使用CPU进行训练，建议检查以下项目:")
        print("1. 是否安装了NVIDIA显卡驱动?")
        print("2. 是否安装了对应版本的CUDA工具包?")
        print("3. 是否安装了支持CUDA的PyTorch版本?")
        print("提示: 访问 https://pytorch.org 获取GPU版本安装指南")

if __name__ == "__main__":
    check_gpu_availability()