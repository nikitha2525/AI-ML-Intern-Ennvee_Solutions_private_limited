# 🎥 Face Recognition Attendance System (CNN + Computer Vision)

A real-time, webcam-based attendance system that uses **Convolutional Neural Networks (CNN)** and **Computer Vision** to automatically detect, recognize, and mark attendance for individuals — eliminating manual roll calls and buddy-punching.

---

## 📌 Overview

This project captures a live video feed from a webcam, detects faces frame-by-frame, and matches them against a database of known faces using deep learning-based facial embeddings. Once a match crosses a confidence threshold, attendance is automatically logged with a timestamp — no manual intervention required.

Core pipeline:
1. **Face Detection** — locate face(s) in each webcam frame
2. **Face Alignment & Preprocessing** — crop, resize, normalize
3. **CNN-based Feature Extraction** — generate a facial embedding vector
4. **Recognition/Matching** — compare embedding against stored known-face embeddings
5. **Attendance Logging** — mark present with name + timestamp if confidence passes threshold

---

## ✨ Features

- 🎥 **Real-time webcam face detection** using OpenCV (Haar Cascade / DNN face detector)
- 🧠 **CNN-based face recognition** for robust identity matching (lighting/angle tolerant)
- 🆔 **Multi-face detection** — recognizes multiple people in a single frame
- 📝 **Automatic attendance logging** with date, time, and confidence score
- 🚫 **Duplicate-entry prevention** — won't mark the same person twice in one session
- 📊 **Attendance dashboard/export** — view or export logs (CSV/Excel)
- 🖼️ **New face enrollment** — register new individuals via webcam capture
- ⚡ **PyTorch-powered inference** for fast, GPU-accelerated recognition (CPU fallback supported)

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Computer Vision | OpenCV |
| Deep Learning Framework | PyTorch |
| Face Detection | Haar Cascade / OpenCV DNN Face Detector |
| Face Recognition | CNN (custom or pretrained, e.g., FaceNet-style embedding model) |
| Data Storage | CSV / |
| Face Embeddings Store | NumPy arrays / Pickle |


---

## 🧠 How It Works

```
Webcam Feed
    │
    ▼
Frame Capture (OpenCV) ──► Face Detection (Haar/DNN) ──► Crop & Align Face
    │
    ▼
Preprocessing (resize, normalize, grayscale/RGB) 
    │
    ▼
CNN Feature Extractor ──► 128/512-D Face Embedding
    │
    ▼
Compare vs Known Embeddings (Euclidean/Cosine Distance)
    │
    ▼
Match Found? ──Yes──► Log Attendance (Name, Timestamp, Confidence)
    │
   No
    │
    ▼
Label as "Unknown"
```

**CNN Architecture (typical):**
- Input: 100×100 or 160×160 face crop (RGB)
- Convolutional + MaxPooling layers for feature extraction
- Fully connected embedding layer (128 or 512 dimensions)
- Trained/fine-tuned using triplet loss or classification loss on labeled face dataset
- Inference outputs a feature vector compared via distance metric to enrolled faces

---

## 📁 Project Structure

```
face-attendance-system/
│
├── main.py                        # Entry point — launches webcam attendance loop
├── enroll.py                      # Script to register/enroll new faces
├── requirements.txt                # Python dependencies
│
├── models/
│   ├── cnn_model.py                # CNN architecture definition (PyTorch)
│   ├── face_recognition_model.pth  # Trained model weights
│   └── face_detector.xml           # Haar Cascade / DNN detector files
│
├── utils/
│   ├── preprocess.py               # Face cropping, resizing, normalization
│   ├── embeddings.py               # Generate & compare face embeddings
│   └── logger.py                   # Attendance logging logic
│
├── data/
│   ├── known_faces/                # Enrolled face images per person
│   ├── embeddings.pkl              # Stored embeddings for known faces
│   └── attendance_log.csv          # Daily attendance records
│
├── dashboard/                      # Optional Flask/Streamlit UI
│   └── app.py
│
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/nikitha2525/face-attendance-cnn.git
cd face-attendance-cnn
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt** should include:
```
opencv-python
torch
torchvision
numpy
pandas
scikit-learn
```

### 4. Enroll known faces
```bash
python enroll.py --name "Nikitha" --samples 20
```
This captures multiple webcam frames of the person and stores their face embedding.

### 5. Run the attendance system
```bash
python main.py
```
The webcam feed opens, detects faces, and logs attendance automatically to `data/attendance_log.csv`.

---

## 🧪 Example Usage

```python
from utils.preprocess import detect_and_crop_face
from utils.embeddings import get_embedding, match_face
from utils.logger import mark_attendance
import cv2

frame = cv2.imread("sample_frame.jpg")
face = detect_and_crop_face(frame)

if face is not None:
    embedding = get_embedding(face)
    name, confidence = match_face(embedding, threshold=0.6)

    if name != "Unknown":
        mark_attendance(name, confidence)
        print(f"Attendance marked for {name} (confidence: {confidence:.2f})")
    else:
        print("Face not recognized.")
```

---

## 📊 Attendance Log Format

| Name | Date | Time | Confidence |
|---|---|---|---|
| Nikitha | 2026-07-22 | 09:03:15 | 0.91 |
| Ashok | 2026-07-22 | 09:04:02 | 0.87 |

---

## 🚀 Future Enhancements

- [ ] Liveness detection to prevent spoofing via photos/videos
- [ ] Mask-aware face recognition
- [ ] Cloud sync of attendance logs (Firebase/PostgreSQL)
- [ ] Admin dashboard with attendance analytics & charts
- [ ] Edge deployment on Raspberry Pi with lightweight CNN model
- [ ] Email/SMS alerts for absentees

---

## 👩‍💻 Author

**Nikitha**
B.Tech AI & Data Science | PSR Engineering College
[GitHub: nikitha2525](https://github.com/nikitha2525)

---

