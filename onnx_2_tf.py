import onnx
from onnx_tf.backend import prepare

# 加载 ONNX 模型
onnx_model = onnx.load("mobilenetv2_custom.onnx")

# 转换为 TensorFlow 格式（SavedModel）
tf_rep = prepare(onnx_model)  # 自动生成 TensorFlow 计算图
tf_rep.export_graph("tf_model")  # 输出为 SavedModel 格式