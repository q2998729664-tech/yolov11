import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from ultralytics import YOLO


class YOLOGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv11 目标检测系统")
        self.root.geometry("1200x700")

        # 变量
        self.model_path = tk.StringVar(value="best.pt")  # 默认模型
        self.conf_thres = tk.DoubleVar(value=0.25)
        self.iou_thres = tk.DoubleVar(value=0.45)
        self.stop_flag = False  # 用于停止摄像头/视频检测
        self.cap = None  # 视频捕获对象
        self.after_id = None  # 用于取消 after 循环

        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧控制面板
        left_frame = ttk.LabelFrame(main_frame, text="模型设置", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 模型选择
        ttk.Label(left_frame, text="选择模型:").grid(row=0, column=0, sticky=tk.W, pady=5)
        model_entry = ttk.Entry(left_frame, textvariable=self.model_path, width=20)
        model_entry.grid(row=0, column=1, pady=5, padx=5)
        ttk.Button(left_frame, text="浏览", command=self.browse_model).grid(row=0, column=2, pady=5)

        # 置信度阈值
        ttk.Label(left_frame, text="置信度阈值:").grid(row=1, column=0, sticky=tk.W, pady=5)
        conf_scale = ttk.Scale(
            left_frame, from_=0.0, to=1.0, variable=self.conf_thres, orient=tk.HORIZONTAL, length=150
        )
        conf_scale.grid(row=1, column=1, columnspan=2, pady=5, sticky=tk.W)
        self.conf_label = ttk.Label(left_frame, text=f"{self.conf_thres.get():.2f}")
        self.conf_label.grid(row=1, column=3, padx=5)
        conf_scale.configure(command=lambda x: self.conf_label.config(text=f"{float(x):.2f}"))

        # IoU阈值
        ttk.Label(left_frame, text="IoU值:").grid(row=2, column=0, sticky=tk.W, pady=5)
        iou_scale = ttk.Scale(left_frame, from_=0.0, to=1.0, variable=self.iou_thres, orient=tk.HORIZONTAL, length=150)
        iou_scale.grid(row=2, column=1, columnspan=2, pady=5, sticky=tk.W)
        self.iou_label = ttk.Label(left_frame, text=f"{self.iou_thres.get():.2f}")
        self.iou_label.grid(row=2, column=3, padx=5)
        iou_scale.configure(command=lambda x: self.iou_label.config(text=f"{float(x):.2f}"))

        # 功能按钮
        ttk.Label(left_frame, text="功能", font=("Arial", 10, "bold")).grid(row=3, column=0, columnspan=3, pady=(15, 5))
        ttk.Button(left_frame, text="图片检测", command=self.detect_image).grid(
            row=4, column=0, columnspan=3, pady=5, sticky=tk.EW
        )
        ttk.Button(left_frame, text="视频检测", command=self.detect_video).grid(
            row=5, column=0, columnspan=3, pady=5, sticky=tk.EW
        )
        ttk.Button(left_frame, text="摄像头检测", command=self.detect_webcam).grid(
            row=6, column=0, columnspan=3, pady=5, sticky=tk.EW
        )
        ttk.Button(left_frame, text="停止检测", command=self.stop_detection).grid(
            row=7, column=0, columnspan=3, pady=5, sticky=tk.EW
        )

        # 右侧显示区域
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 图像显示
        self.image_label = ttk.Label(right_frame, text="检测结果", relief=tk.SUNKEN, anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # 检测结果表格
        columns = ("类别", "置信度", "位置 (x, y, w, h)")
        self.tree = ttk.Treeview(right_frame, columns=columns, show="headings", height=8)
        self.tree.heading("类别", text="类别")
        self.tree.heading("置信度", text="置信度")
        self.tree.heading("位置 (x, y, w, h)", text="位置 (x, y, w, h)")
        self.tree.column("类别", width=100)
        self.tree.column("置信度", width=80)
        self.tree.column("位置 (x, y, w, h)", width=250)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # 状态栏
        self.status_bar = ttk.Label(self.root, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def browse_model(self):
        filename = filedialog.askopenfilename(filetypes=[("PyTorch模型", "*.pt")])
        if filename:
            self.model_path.set(filename)

    def update_status(self, text):
        self.status_bar.config(text=text)
        self.root.update_idletasks()

    def load_model(self):
        """加载模型，如果失败则弹出错误."""
        try:
            model = YOLO(self.model_path.get())
            self.update_status("模型加载成功")
            return model
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败：{e!s}")
            return None

    def detect_image(self):
        model = self.load_model()
        if model is None:
            return
        file_path = filedialog.askopenfilename(filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return
        self.update_status("正在检测图片...")
        # 运行检测
        results = model(file_path, conf=self.conf_thres.get(), iou=self.iou_thres.get())[0]
        # 绘制结果
        img = results.plot()  # BGR numpy array
        self.display_image(img)
        self.update_detection_table(results)
        self.update_status("图片检测完成")

    def display_image(self, cv_img):
        """将OpenCV图像（BGR）显示在Label上."""
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_img)
        # 缩放以适应显示区域（保持宽高比）
        max_width = self.image_label.winfo_width() or 600
        max_height = self.image_label.winfo_height() or 400
        pil_img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=pil_img)
        self.image_label.config(image=imgtk)
        self.image_label.image = imgtk  # 保持引用

    def update_detection_table(self, results):
        """清空并更新检测结果表格."""
        for item in self.tree.get_children():
            self.tree.delete(item)
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = results.names[cls_id]
                conf = float(box.conf[0])
                xywh = box.xywh[0].tolist()  # [x, y, w, h]
                pos_str = f"({xywh[0]:.2f}, {xywh[1]:.2f}, {xywh[2]:.2f}, {xywh[3]:.2f})"
                self.tree.insert("", tk.END, values=(cls_name, f"{conf:.2f}", pos_str))

    def detect_video(self):
        model = self.load_model()
        if model is None:
            return
        file_path = filedialog.askopenfilename(filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
        if not file_path:
            return
        self.stop_flag = False
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开视频文件")
            return
        self.update_status("正在检测视频...")
        self.play_video(model)

    def detect_webcam(self):
        model = self.load_model()
        if model is None:
            return
        self.stop_flag = False
        self.cap = cv2.VideoCapture(0)  # 默认摄像头
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            return
        self.update_status("正在检测摄像头画面...")
        self.play_video(model)

    def play_video(self, model):
        """循环读取视频帧并显示（在新窗口中）."""
        if self.stop_flag or self.cap is None:
            self.cap.release()
            self.update_status("检测已停止")
            return
        ret, frame = self.cap.read()
        if not ret:
            # 视频结束
            self.cap.release()
            self.update_status("视频检测结束")
            return
        # 检测
        results = model(frame, conf=self.conf_thres.get(), iou=self.iou_thres.get())[0]
        annotated_frame = results.plot()
        # 显示
        cv2.imshow("YOLO Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # 按q停止
            self.stop_detection()
        # 继续下一帧
        self.after_id = self.root.after(10, lambda: self.play_video(model))

    def stop_detection(self):
        self.stop_flag = True
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.update_status("检测已停止")


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOGUI(root)
    root.mainloop()
