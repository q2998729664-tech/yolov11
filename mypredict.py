from ultralytics import YOLO
model = YOLO(r"C:\pythoncode\yolocode\ultralytics-8.3.163\results\yolo11m4\weights\best.pt")
model.predict(
    source=r"C:\Users\29987\Downloads\QQ2026217-225016.mp4",
    save = False,
    show = True,
    conf = 0.05,
    iou = 0.9,
    max_det = 99999,
)
