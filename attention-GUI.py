import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import os
import sys
import csv
from datetime import datetime
import time
from ultralytics import YOLO
import csv
from datetime import datetime

# ========== 专注度规则函数 ==========
def classify_attention(ratios):
    """
    根据情绪比例规则判断专注度等级
    ratios: dict, 包含情绪名称到占比的字典（0~1）
    返回: '高专注' / '中专注' / '低专注'
    """
    focused = ratios.get('focused', 0.0)
    happy = ratios.get('happy', 0.0)
    confused = ratios.get('confused', 0.0)
    bored = ratios.get('bored', 0.0)
    irritated = ratios.get('irritated', 0.0)
    resistant = ratios.get('resistant', 0.0)

    positive = focused + happy
    negative = irritated + resistant
    neutral = confused + bored

    # 阈值可根据实际调整
    if positive > 0.6 and negative < 0.1:
        return "高专注"
    elif neutral > 0.4 or (0.1 <= negative <= 0.3):
        return "中专注"
    else:
        return "低专注"


class YOLOAttentionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv11 课堂专注度分析系统")
        self.root.geometry("1400x800")

        # 变量
        self.model_path = tk.StringVar(value="best.pt")
        self.conf_thres = tk.DoubleVar(value=0.25)
        self.iou_thres = tk.DoubleVar(value=0.7)
        self.stop_flag = False          # 用于停止视频/摄像头检测
        self.cap = None
        self.after_id = None
        self.model = None               # 模型实例，懒加载

        # 情绪名称列表（必须与训练一致）
        self.emotion_names = ['focused', 'confused', 'happy', 'bored', 'irritated', 'resistant']
        # 统计变量（累计计数）
        self.total_detections = 0
        self.emotion_counts = {name: 0 for name in self.emotion_names}

        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧控制面板
        left_frame = ttk.LabelFrame(main_frame, text="模型设置", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))

        # 模型选择
        ttk.Label(left_frame, text="选择模型:").grid(row=0, column=0, sticky=tk.W, pady=5)
        model_entry = ttk.Entry(left_frame, textvariable=self.model_path, width=20)
        model_entry.grid(row=0, column=1, pady=5, padx=5)
        ttk.Button(left_frame, text="浏览", command=self.browse_model).grid(row=0, column=2, pady=5)

        # 置信度阈值
        ttk.Label(left_frame, text="置信度阈值:").grid(row=1, column=0, sticky=tk.W, pady=5)
        conf_scale = ttk.Scale(left_frame, from_=0.0, to=1.0, variable=self.conf_thres,
                                orient=tk.HORIZONTAL, length=150)
        conf_scale.grid(row=1, column=1, columnspan=2, pady=5, sticky=tk.W)
        self.conf_label = ttk.Label(left_frame, text=f"{self.conf_thres.get():.2f}")
        self.conf_label.grid(row=1, column=3, padx=5)
        conf_scale.configure(command=lambda x: self.conf_label.config(text=f"{float(x):.2f}"))

        # IoU阈值
        ttk.Label(left_frame, text="IoU值:").grid(row=2, column=0, sticky=tk.W, pady=5)
        iou_scale = ttk.Scale(left_frame, from_=0.0, to=1.0, variable=self.iou_thres,
                               orient=tk.HORIZONTAL, length=150)
        iou_scale.grid(row=2, column=1, columnspan=2, pady=5, sticky=tk.W)
        self.iou_label = ttk.Label(left_frame, text=f"{self.iou_thres.get():.2f}")
        self.iou_label.grid(row=2, column=3, padx=5)
        iou_scale.configure(command=lambda x: self.iou_label.config(text=f"{float(x):.2f}"))

        # 功能按钮
        ttk.Label(left_frame, text="功能", font=('Arial', 10, 'bold')).grid(row=3, column=0, columnspan=3, pady=(15,5))
        ttk.Button(left_frame, text="图片检测", command=self.detect_image).grid(row=4, column=0, columnspan=3, pady=5, sticky=tk.EW)
        ttk.Button(left_frame, text="视频检测", command=self.detect_video).grid(row=5, column=0, columnspan=3, pady=5, sticky=tk.EW)
        ttk.Button(left_frame, text="摄像头检测", command=self.detect_webcam).grid(row=6, column=0, columnspan=3, pady=5, sticky=tk.EW)
        ttk.Button(left_frame, text="停止检测", command=self.stop_detection).grid(row=7, column=0, columnspan=3, pady=5, sticky=tk.EW)

        # 右侧显示区域
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 图像显示区域
        self.image_label = ttk.Label(right_frame, text="检测结果", relief=tk.SUNKEN, anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=(0,10))

        # 统计信息区域
        stats_frame = ttk.LabelFrame(right_frame, text="情绪统计与专注度", padding="5")
        stats_frame.pack(fill=tk.X, pady=(0,10))

        # 创建一个网格用于显示各类情绪占比
        self.ratio_vars = {}
        row = 0
        for i, name in enumerate(self.emotion_names):
            lbl = ttk.Label(stats_frame, text=f"{name}:")
            lbl.grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(value="0.00%")
            self.ratio_vars[name] = var
            ttk.Label(stats_frame, textvariable=var).grid(row=row, column=1, sticky=tk.W, padx=5)
            row += 1

        # 专注度等级
        ttk.Label(stats_frame, text="专注度等级:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.attention_var = tk.StringVar(value="未知")
        ttk.Label(stats_frame, textvariable=self.attention_var, font=('Arial', 10, 'bold')).grid(row=row, column=1, sticky=tk.W, padx=5)
        # 添加保存结果按钮
        ttk.Button(stats_frame, text="保存结果", command=self.save_results).grid(row=row + 1, column=0, columnspan=2,
                                                                                 pady=10)
        # 检测结果表格（可选，保留但可不显示位置）
        columns = ('类别', '置信度')   # 省略位置列，保持简洁
        self.tree = ttk.Treeview(right_frame, columns=columns, show='headings', height=6)
        self.tree.heading('类别', text='类别')
        self.tree.heading('置信度', text='置信度')
        self.tree.column('类别', width=100)
        self.tree.column('置信度', width=80)
        self.tree.pack(fill=tk.BOTH, expand=True)



        # 状态栏
        self.status_bar = ttk.Label(self.root, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def browse_model(self):
        filename = filedialog.askopenfilename(filetypes=[("PyTorch模型", "*.pt")])
        if filename:
            self.model_path.set(filename)

    def load_model(self):
        """加载模型，如果失败则弹出错误"""
        if self.model is not None:
            return self.model
        try:
            self.model = YOLO(self.model_path.get())
            self.update_status("模型加载成功")
            return self.model
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败：{str(e)}")
            return None

    def update_status(self, text):
        self.status_bar.config(text=text)
        self.root.update_idletasks()

    def reset_stats(self):
        """重置统计计数器"""
        self.total_detections = 0
        for name in self.emotion_names:
            self.emotion_counts[name] = 0
        # 清空显示
        for name, var in self.ratio_vars.items():
            var.set("0.00%")
        self.attention_var.set("未知")

    def update_stats_display(self):
        """根据当前累计计数更新界面上的占比和专注度"""
        if self.total_detections == 0:
            return
        ratios = {}
        for name in self.emotion_names:
            ratio = self.emotion_counts[name] / self.total_detections
            ratios[name] = ratio
            self.ratio_vars[name].set(f"{ratio:.2%}")
        # 计算专注度
        level = classify_attention(ratios)
        self.attention_var.set(level)

    def process_frame(self, frame, update_stats=True, display_table=False):
        """
        对单帧进行检测，可选是否更新统计信息和表格
        返回标注后的图像
        """
        results = self.model(frame, conf=self.conf_thres.get(), iou=self.iou_thres.get(), verbose=False)[0]
        annotated = results.plot()

        if update_stats:
            # 更新统计计数
            if results.boxes is not None:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = results.names[cls_id]  # 获取情绪名称
                    if cls_name in self.emotion_names:
                        self.emotion_counts[cls_name] += 1
                        self.total_detections += 1

        if display_table:
            # 更新表格（可选，只显示当前帧的检测结果）
            for item in self.tree.get_children():
                self.tree.delete(item)
            if results.boxes is not None:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = results.names[cls_id]
                    conf = float(box.conf[0])
                    self.tree.insert('', tk.END, values=(cls_name, f"{conf:.2f}"))

        return annotated
    # # 不缩放尺寸
    # def display_image(self, cv_img):
    #     """将OpenCV图像（BGR）显示在Label上"""
    #     cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    #     pil_img = Image.fromarray(cv_img)
    #     # 缩放以适应显示区域
    #     max_width = self.image_label.winfo_width() or 600
    #     max_height = self.image_label.winfo_height() or 400
    #     pil_img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    #     imgtk = ImageTk.PhotoImage(image=pil_img)
    #     self.image_label.config(image=imgtk)
    #     self.image_label.image = imgtk

    #缩放尺寸
    def display_image(self, cv_img):
        """
        将 OpenCV 图像（BGR）显示在 GUI 的 Label 上，
        缩放至固定尺寸（例如 800x600），保持宽高比，多余部分填充黑色。
        """
        # ===== 固定显示尺寸（可根据需要调整） =====
        DISPLAY_WIDTH = 400
        DISPLAY_HEIGHT = 300
        # ========================================

        # BGR 转 RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # 计算等比例缩放后的尺寸
        ratio = min(DISPLAY_WIDTH / pil_img.width, DISPLAY_HEIGHT / pil_img.height)
        new_w = int(pil_img.width * ratio)
        new_h = int(pil_img.height * ratio)

        # 缩放图像
        resized_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 创建固定大小的背景图像（黑色）
        background = Image.new('RGB', (DISPLAY_WIDTH, DISPLAY_HEIGHT), (0, 0, 0))
        # 计算粘贴位置（居中）
        offset = ((DISPLAY_WIDTH - new_w) // 2, (DISPLAY_HEIGHT - new_h) // 2)
        background.paste(resized_img, offset)

        # 转换为 Tkinter 可显示的格式
        imgtk = ImageTk.PhotoImage(image=background)
        self.image_label.config(image=imgtk)
        self.image_label.image = imgtk  # 保持引用，防止被垃圾回收



    # ---------- 图片检测 ----------
    def detect_image(self):
        model = self.load_model()
        if model is None:
            return
        file_path = filedialog.askopenfilename(filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return
        self.reset_stats()
        self.update_status("正在检测图片...")
        frame = cv2.imread(file_path)
        annotated = self.process_frame(frame, update_stats=True, display_table=True)
        self.display_image(annotated)
        self.update_stats_display()
        self.update_status("图片检测完成")

    # ---------- 视频/摄像头检测 ----------
    def detect_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
        if not file_path:
            return
        self.start_video_capture(file_path)

    def detect_webcam(self):
        self.start_video_capture(0)  # 0 表示默认摄像头

    def start_video_capture(self, source):
        model = self.load_model()
        if model is None:
            return
        self.stop_flag = False
        self.reset_stats()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开视频源")
            return
        self.update_status("正在检测...")
        # 启动新线程进行视频处理
        thread = threading.Thread(target=self.video_loop, daemon=True)
        thread.start()

    def video_loop(self):
        """视频处理循环，在后台线程中运行，将帧显示在 GUI 上"""
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        # 如果没有有效的帧率，默认使用 30 FPS 的延时
        delay = 1 / fps if fps > 0 else 0.03

        while not self.stop_flag:
            ret, frame = self.cap.read()
            if not ret:
                break
            # 处理帧（更新统计信息，但不更新表格以免频繁刷新）
            annotated = self.process_frame(frame, update_stats=True, display_table=False)

            # 在主线程中更新 GUI 显示
            self.root.after(0, self.display_image, annotated)
            self.root.after(0, self.update_stats_display)

            # 简单延时，避免过快消耗 CPU（可根据实际调整）
            time.sleep(delay)

        self.cap.release()
        self.root.after(0, self.video_finished)

    def video_finished(self):
        """视频处理结束后的回调"""
        self.update_stats_display()
        self.update_status("视频检测完成")

    def stop_detection(self):
        self.stop_flag = True
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.update_status("检测已停止")

    def __del__(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def save_results(self):
            """将当前统计结果保存到文件"""
            if self.total_detections == 0:
                 messagebox.showwarning("警告", "没有检测数据可保存")
                 return

            # 创建 results 文件夹
            save_dir = r"C:\Users\29987\Desktop\detection_results"
            os.makedirs(save_dir, exist_ok=True)

            # 生成文件名（使用当前时间）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"result_{timestamp}.csv")

            # 准备数据
            data = {
                "timestamp": timestamp,
                "total_detections": self.total_detections,
            }
            # 添加各类情绪占比
            ratios = {}
            for name in self.emotion_names:
                ratio = self.emotion_counts[name] / self.total_detections
                ratios[name] = ratio
                data[f"{name}_count"] = self.emotion_counts[name]
                data[f"{name}_ratio"] = round(ratio, 4)

            # 专注度等级
            level = self.attention_var.get()
            data["attention_level"] = level

            # 写入 CSV 文件
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                # 写表头和数据
                writer.writerow(data.keys())
                writer.writerow(data.values())

            # 同时生成一个易读的 TXT 文件（可选）
            txt_filename = os.path.join(save_dir, f"result_{timestamp}.txt")
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(f"检测时间：{timestamp}\n")
                f.write(f"总检测人脸数：{self.total_detections}\n\n")
                f.write("情绪占比：\n")
                for name in self.emotion_names:
                    f.write(f"  {name}: {ratios[name]:.2%}\n")
                f.write(f"\n专注度等级：{level}\n")

            messagebox.showinfo("保存成功", f"结果已保存到\n{filename}\n以及同目录的 txt 文件")

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOAttentionGUI(root)
    root.mainloop()