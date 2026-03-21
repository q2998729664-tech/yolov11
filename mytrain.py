# from ultralytics import YOLO
# ms = [
#     'yolo11m','yolo12m',
# ]
# if __name__ == "__main__":
#     for m in ms:
#         model = YOLO(m + ".pt")
#         model.train(
#             data = r"emotion.yaml",
#             epochs = 100,
#             imgsz = 640,
#             batch = -1,
#             cache = 'ram',
#             workers = 2,
#             multi_scale = True,
#             scale = 0.5,
#             copy_paste = 0.3,
#             mixup = 0.2,
#             patience = 20,
#             augment = True,
#             project="results",
#             name=m,
#
#         )

from ultralytics import YOLO

if __name__ == "__main__":
    # 正确的路径写法，对应你文件的位置
    # 只需要修改这一行，前面加r
    model = YOLO(r"ultralytics\cfg\models\11\yolo11-CBAMRFA.yaml").load("yolo11m.pt")
    model.train(
        data=r"emotion.yaml",
        epochs=150,
        imgsz=640,
        batch=-1,
        cache="ram",
        workers=2,
        multi_scale=True,
        scale=0.5,
        copy_paste=0.3,
        patience=20,
        mixup=0.2,
        augment=True,
        project="results",
        name="yolo11m_rfa",
    )
