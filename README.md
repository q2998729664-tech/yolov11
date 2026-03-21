YOLOv11-RSCN: Real-time Classroom Attention Analysis via Fine-grained Emotion Recognition
This repository implements a YOLOv11-based end-to-end system for fine-grained emotion recognition and classroom attention analysis. The model integrates two attention mechanisms — RFAConv and CBAM — along with Focal Loss to tackle class imbalance, achieving state-of-the-art performance on a real‑world classroom emotion dataset.

https://img.shields.io/badge/Python-3.9+-blue.svg
https://img.shields.io/badge/PyTorch-2.0+-orange.svg
https://img.shields.io/badge/License-AGPL--3.0-green.svg

📌 Overview
Traditional classroom attention assessment relies on subjective observation, which is inefficient and fails to capture student engagement in detail. This project develops a fully automated system that:

Detects faces and recognizes six fine-grained emotions: focused, confused, happy, bored, irritated, resistant.

Quantifies attention levels (high / medium / low) using a machine learning model trained on emotion distributions.

Runs in real time (≥78 FPS on RTX 5060) and provides a graphical interface for classroom monitoring.

✨ Key Features
Attention-Enhanced Backbone

RFAConv (Receptive‑Field Attention Convolution): dynamically assigns pixel‑level weights inside each receptive field, capturing subtle facial expressions.

CBAM (Convolutional Block Attention Module): refines features along channel and spatial dimensions to highlight discriminative regions.

Class‑imbalance Handling

Focal Loss reduces the contribution of easy examples (e.g., “focused”) and forces the model to focus on rare emotions (e.g., “irritated”, “resistant”).

Attention‑to‑Concentration Mapping

Pearson correlation analysis reveals strong relationships (e.g., “focused” ↔ high attention, “bored” ↔ low attention).

A random forest model converts emotion ratios into attention levels with 86.0% accuracy.

Real‑time Graphical Interface

Supports image, video, and webcam input.

Live bounding boxes, emotion labels, and attention level display.

One‑click result saving (CSV/TXT).

🚀 Performance Highlights
Metric	Baseline YOLOv11m	Ours (YOLOv11‑RSCN)
mAP50	87.3%	91.5% (+4.2%)
mAP50:95	66.5%	70.8%
Parameters	20.1M	20.4M (+1.5%)
Inference Speed	82 FPS	78 FPS (RTX 5060)
Attention Accuracy (RF)	–	86.0%
📁 Repository Structure
text
.
├── configs/
│   └── yolo11-CBAMRFA.yaml          # Model configuration with RFAConv & CBAM
├── models/
│   ├── conv.py                      # Custom modules (RFAConv, CBAM, etc.)
│   └── attention.py                 # Channel & Spatial attention implementations
├── datasets/
│   └── classroom_emotion.yaml       # Dataset configuration (paths, classes)
├── scripts/
│   ├── train.py                     # Training script
│   ├── val.py                       # Validation script
│   └── predict.py                   # Inference script
├── gui/
│   └── attention_gui.py             # Tkinter‑based real‑time demo
├── results/                         # Trained weights and logs
├── requirements.txt
└── README.md
🛠️ Installation
Clone the repository

bash
cd yolov11-classroom-attention
Create a virtual environment (recommended)

bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
Install dependencies

bash
pip install -r requirements.txt
The requirements.txt includes:

text
ultralytics==8.3.163
torch>=2.0.0
torchvision>=0.15.0
opencv-python
pillow
numpy
scikit-learn
matplotlib
einops
📊 Dataset
We use the Chinese University Real Classroom Emotion Dataset [1] as the primary source. It contains 5,527 images annotated with five emotions (neutral, happy, surprised, confused, bored). After mapping to our six‑class scheme and supplementing with DIPSER data, we obtain ~6,500 images.

The dataset is organized in YOLO format:

text
datasets/classroom_emotion/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
A sample dataset.yaml is provided. You can replace the path with your own data location.

🏋️ Training
To train the model from scratch with the proposed improvements:

bash
python scripts/train.py --data datasets/classroom_emotion.yaml --cfg configs/yolo11-CBAMRFA.yaml --weights yolo11m.pt --epochs 150 --batch 16 --imgsz 640
Key training arguments:

--cfg: YAML model definition (our custom architecture).

--weights: pretrained YOLOv11m weights for transfer learning.

--epochs: 150 with early stopping patience 20.

--batch: adjust according to GPU memory (use --batch -1 for automatic selection).

Training logs and checkpoints are saved in runs/train/.

🔍 Evaluation
Evaluate the trained model on the test set:

bash
python scripts/val.py --weights runs/train/exp/weights/best.pt --data datasets/classroom_emotion.yaml --imgsz 640
The script will output mAP50, mAP50:95, and per‑class precision/recall.

🖥️ GUI Demo
Launch the real‑time attention analysis interface:

bash
python gui/attention_gui.py
The GUI allows you to:

Load a custom model (default: best.pt).

Adjust confidence and IoU thresholds.

Process images, videos, or live webcam.

View emotion ratios and attention level updates in real time.

Save results as CSV/TXT for further analysis.

📈 Results & Ablation Study
We conducted extensive ablation experiments on the classroom dataset. All models were trained for 150 epochs with identical hyperparameters.

Model	mAP50 (%)	mAP50:95 (%)	Params (M)	FPS
YOLOv11m (baseline)	87.3	66.5	20.1	82
+ RFAConv	89.2	68.4	20.3	79
+ CBAM	88.9	67.8	20.2	80
+ RFAConv + CBAM	90.4	69.6	20.4	78
+ Focal Loss (full)	91.5	70.8	20.4	78
In challenging scenarios (occlusion, backlight, distant faces), the proposed model exhibits significantly improved robustness, e.g., +16.4% mAP under heavy occlusion.


🙏 Acknowledgements
Ultralytics YOLOv11 for the excellent detection framework.

The authors of RFAConv and CBAM for their inspiring work.
