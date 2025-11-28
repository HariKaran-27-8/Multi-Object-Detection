# YOLOv8 + DeepSORT Real-Time Object Detection & Tracking

This project implements a real-time object detection and multi-object tracking system using YOLOv8 for detection and DeepSORT for assigning consistent tracking IDs across frames.

## 🚀 Features
- Real-time object detection  
- Multi-object tracking (DeepSORT)  
- Unique IDs for each tracked object  
- Works on webcam or video files  
- Supports any YOLOv8 model (n/s/m/l/x)  
- Optional output video saving  

## 🧠 How It Works

### YOLOv8  
Detects objects in each frame and returns:
- Bounding box  
- Class label  
- Confidence score  

### DeepSORT  
Uses YOLO detections to perform:
- Tracking  
- Re-identification  
- Motion prediction  
- Maintaining IDs after occlusion  

The final output displays:
- Bounding boxes  
- Class names  
- Tracking IDs  
- FPS counter  

## 📂 Project Structure
```
project/
│
├── main.py
├── requirements.txt
├── yolov8n.pt
└── (optional video files)
```

## 🔧 Installation

### 1. Create a virtual environment
```
python -m venv venv
```

### 2. Activate the environment
**Windows CMD**
```
venv\Scripts\activate
```

**PowerShell**
```
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

## ▶️ Running the Project

### Use Webcam
```
python main.py --source 0
```

### Use Video File
```
python main.py --source "highway.mp4"
```

### Full Path Example
```
python main.py --source "D:\data\videos\highway.mp4"
```

### Save Output Video
```
python main.py --source highway.mp4 --output output.mp4
```

## 📝 Arguments
| Argument | Description |
|---------|-------------|
| `--source` | Webcam (0) or video file path |
| `--model` | YOLOv8 model file (default yolov8n.pt) |
| `--conf` | Confidence threshold |
| `--iou` | IOU threshold |
| `--show` | Show video window |
| `--output` | Save output video |

## 💡 Applications
- Traffic monitoring  
- Surveillance  
- Vehicle counting  
- Crowd analytics  
- Smart city systems  
- Autonomous navigation  

## 🙌 Acknowledgements
- Ultralytics YOLOv8  
- DeepSORT-Realtime  
- OpenCV  
