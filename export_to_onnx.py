import torch
import torchvision.models as models

# 1. 加载模型并修改分类层
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(1280, 15)  # 改为15类输出

# 2. 加载.pth文件（安全模式）
checkpoint = torch.load("best_model.pth", weights_only=True)
model.load_state_dict(checkpoint)
model.eval()

# 3. 导出ONNX
dummy_input = torch.randn(1, 3, 224, 224)  # 假设输入为224x224
torch.onnx.export(
    model,
    dummy_input,
    "mobilenetv2_custom.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=11
)

print("导出成功！")