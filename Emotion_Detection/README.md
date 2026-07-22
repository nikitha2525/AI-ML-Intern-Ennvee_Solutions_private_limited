# 😊😢😠 Facial Emotion Detection (CNN + PyTorch + Gradio)

A deep learning web application that detects human emotions in real time by analyzing facial expressions from an image or live webcam feed. Built with a **Convolutional Neural Network (CNN)** in **PyTorch**, wrapped in an interactive **Gradio** interface, and deployed publicly on **Hugging Face Spaces**.

🔗 **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/nikitha2525/emotion-detection) <!-- update with actual link -->

---

## 📌 Overview

This project classifies a human face into one of several emotion categories using a CNN trained on facial expression image data. Users can either **upload an image** or use their **webcam** directly in the browser, and the model returns the predicted emotion along with confidence scores — all served through a lightweight Gradio UI hosted on Hugging Face.

**Supported Emotion Classes:**
- 😀 Happy
- 😢 Sad
- 😠 Angry
- 😨 Fear
- 😲 Surprise


---

## ✨ Features

- 🧠 **Custom CNN model** trained in PyTorch for facial expression classification
- 🎥 **Live webcam inference** directly inside the Gradio interface
- 🖼️ **Image upload support** for static photo emotion detection
- 📊 **Confidence scores** displayed for all emotion classes, not just the top prediction
- ⚡ **Real-time / near real-time inference** optimized for CPU (Hugging Face free tier)
- 🌐 **One-click public deployment** via Hugging Face Spaces (Gradio SDK)
- 🔍 **Face detection preprocessing** (OpenCV/Haar Cascade) before classification for better accuracy

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning Framework | PyTorch |
| Model Architecture | Custom CNN (Conv2D + BatchNorm + MaxPool + FC layers) |
| Face Detection | OpenCV (Haar Cascade / DNN detector) |
| Web Interface | Gradio |
| Deployment | Hugging Face Spaces |


---

## 🧠 How It Works

```
Input Source
 (Webcam Frame / Uploaded Image)
            │
            ▼
   Face Detection (OpenCV)
            │
            ▼
  Crop & Preprocess Face
 (grayscale/RGB, resize 48x48
   or 224x224, normalize)
            │
            ▼
       CNN Model (PyTorch)
            │
            ▼
    Softmax Output Layer
  (7-class probability vector)
            │
            ▼
  Predicted Emotion + Confidence
   (Happy / Sad / Angry
      / Fear /Surprise )
            │
            ▼
     Displayed in Gradio UI
```

**CNN Architecture (typical):**
```
Input (48x48x1 or 224x224x3)
   │
Conv2D(32) → ReLU → BatchNorm → MaxPool
   │
Conv2D(64) → ReLU → BatchNorm → MaxPool
   │
Conv2D(128) → ReLU → BatchNorm → MaxPool
   │
Flatten → Dense(256) → Dropout(0.5)
   │
Dense(7) → Softmax
```


---

## ⚙️ Installation & Setup (Local)

### 1. Clone the repository
```bash
git clone https://github.com/nikitha2525/emotion-detection-cnn.git
cd emotion-detection-cnn
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

**requirements.txt**
```
torch
torchvision
opencv-python
gradio
numpy
pillow
```

### 4. Run locally
```bash
python app.py
```
Gradio will launch a local URL (and optionally a public `.gradio.live` share link) with both **image upload** and **webcam** input tabs.

---

## 🚀 Deploying to Hugging Face Spaces

1. Create a new Space on Hugging Face → select **Gradio** as the SDK
2. Push your repo (must include `app.py` + `requirements.txt` at root)
   ```bash
   git remote add space https://huggingface.co/spaces/<username>/emotion-detection
   git push space main
   ```
3. Hugging Face automatically builds and hosts the app — webcam access works directly in-browser via Gradio's `Image(source="webcam")` component
4. Model weights (`.pth`) can be included in the repo or loaded from the Hugging Face Hub using `huggingface_hub.hf_hub_download`

---

## 🧪 Example Usage (Gradio App Snippet)

```python
import gradio as gr
import torch
from model.emotion_cnn import EmotionCNN
from utils.predict import predict_emotion

model = EmotionCNN()
model.load_state_dict(torch.load("model/emotion_cnn_weights.pth", map_location="cpu"))
model.eval()

def classify_emotion(image):
    label, confidence_scores = predict_emotion(image, model)
    return {label: confidence_scores[label] for label in confidence_scores}

demo = gr.Interface(
    fn=classify_emotion,
    inputs=gr.Image(source="webcam", type="pil", label="Capture your face"),
    outputs=gr.Label(num_top_classes=7, label="Predicted Emotion"),
    title="Facial Emotion Detection",
    description="Detects Happy, Sad, Angry, Fear, Surprise, Neutral, or Disgust from your facial expression."
)

demo.launch()
```



---

## 🚀 Future Enhancements

- [ ] Multi-face emotion detection in a single frame
- [ ] Real-time video stream (frame-by-frame) instead of single-shot capture
- [ ] Transfer learning with a pretrained backbone (ResNet/MobileNet) for higher accuracy
- [ ] Emotion trend tracking over a session (mood timeline)
- [ ] Mobile-friendly / lightweight quantized model for edge deployment

---

## 👩‍💻 Author

**Nikitha**
B.Tech AI & Data Science | PSR Engineering College
[GitHub: nikitha2525](https://github.com/nikitha2525)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
