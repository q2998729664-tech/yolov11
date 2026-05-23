from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"C:\pythoncode\yolocode\ultralytics-8.3.163\results\yolo11l\weights\best.pt")
    model.val(split="test")
