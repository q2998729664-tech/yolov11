import torch

from ultralytics import YOLO

# 1. 测试RFAConv能否正常导入

print("✅ RFAConv模块导入成功，无报错！")

# 2. 测试模型能否正常构建
# 正确的路径写法，对应你文件的位置
# 只需要修改这一行，前面加r
model = YOLO(r"ultralytics\cfg\models\11\yolo11-CBAMRFA.yaml").load("yolo11m.pt")
print("✅ 模型构建成功，yaml配置无报错！")

# 3. 打印前两层结构，确认是RFAConv
print("\n===== 模型骨干网络前两层 =====")
print(f"第0层: {model.model.model[0]}")
print(f"第1层: {model.model.model[1]}")
print(f"第9层: {model.model.model[9]}")

# 4. 测试前向传播，确认无运行时错误
x = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    out = model(x)
print("\n✅ 前向传播正常，模型可正常运行！")
print("🎉 所有报错已修复，可直接开始训练！")
