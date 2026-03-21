import cv2
import numpy as np
from ultralytics import YOLO

# ========== 配置 ==========
MODEL_PATH = r"C:\pythoncode\yolocode\ultralytics-8.3.163\results\yolo11m2\weights\best.pt"           # 你的模型路径
VIDEO_PATH = r"C:\Users\29987\Downloads\QQ2026217-225016.mp4"   # 课堂视频路径
OUTPUT_VIDEO = r"C:\Users\29987\Downloads\output.mp4"   # 输出视频路径（可选，保存带标注的视频）
CONF_THRES = 0.05               # 检测置信度阈值
IOU_THRES = 0.9             # NMS IoU阈值

# 情绪类别名称（必须与训练时一致）
EMOTION_NAMES = ['resistant', 'irritated', 'happy', 'focused', 'bored', 'surprised']
NUM_CLASSES = len(EMOTION_NAMES)

# 规则法阈值（可根据实际情况调整）
POSITIVE_THRES = 0.6    # 高专注时专注+愉悦占比 > 60%
NEGATIVE_LOW_THRES = 0.1  # 高专注时烦躁+抵触占比 < 10%
NEGATIVE_MID_LOW = 0.1    # 中专注时烦躁+抵触的下限
NEGATIVE_MID_HIGH = 0.3   # 中专注时烦躁+抵触的上限
NEUTRAL_THRES = 0.4       # 中专注时困惑+倦怠占比 > 40%
# ==========================

# 1. 加载模型
model = YOLO(MODEL_PATH)

# 2. 打开视频
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 准备输出视频（如果需要）
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 3. 初始化统计变量
total_frames = 0
emotion_counts = {name: 0 for name in EMOTION_NAMES}  # 记录每类情绪出现的总次数（所有帧、所有人脸）

# 4. 逐帧处理
while True:
    ret, frame = cap.read()
    if not ret:
        break
    total_frames += 1

    # 检测
    results = model(frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]

    # 绘制结果（可选）
    annotated_frame = results.plot()
    out.write(annotated_frame)  # 保存到输出视频

    # 统计该帧检测到的每个人脸的情绪
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            emotion_name = results.names[cls_id]  # 注意：results.names 是 {0: 'focused', ...}
            emotion_counts[emotion_name] += 1

# 5. 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

# 6. 计算情绪占比
total_detections = sum(emotion_counts.values())
if total_detections == 0:
    print("未检测到任何人脸，无法分析专注度。")
    exit()

emotion_ratios = {name: count / total_detections for name, count in emotion_counts.items()}
print("情绪占比：")
for name, ratio in emotion_ratios.items():
    print(f"  {name}: {ratio:.2%}")

# 7. 规则法判断专注度等级
def classify_attention(ratios):
    focused = ratios.get('focused', 0.0)
    happy = ratios.get('happy', 0.0)
    confused = ratios.get('confused', 0.0)
    bored = ratios.get('bored', 0.0)
    irritated = ratios.get('irritated', 0.0)
    resistant = ratios.get('resistant', 0.0)

    positive = focused + happy
    negative = irritated + resistant
    neutral = confused + bored

    if positive > POSITIVE_THRES and negative < NEGATIVE_LOW_THRES:
        return "高专注"
    elif neutral > NEUTRAL_THRES or (NEGATIVE_MID_LOW <= negative <= NEGATIVE_MID_HIGH):
        return "中专注"
    else:
        return "低专注"

attention_level = classify_attention(emotion_ratios)
print(f"专注度等级：{attention_level}")