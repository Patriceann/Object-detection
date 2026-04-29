# 📸 Live Object Detection & Tracing
### Activity 3 – Python Streamlit + ML Model  
**Course:** CSE3 | **Institution:** Dr. Emilio B. Espinosa Sr. Memorial State College of Agriculture and Technology

---

## 📌 Description
This is a real-time object detection and tracking web application built using **Streamlit** and **YOLOv8**. The app uses your webcam to detect and track everyday objects like people, phones, bottles, and more — with bounding boxes and labels displayed live on the video feed.

---

## ✨ Features
- 🎥 **Live Webcam Feed** — real-time video stream via WebRTC
- 🔍 **Object Detection** — detects 80 types of objects using YOLOv8
- 🔢 **Object Counting** — counts how many of each object are visible
- 🚨 **Alert Triggers** — triggers a visual alert when specific objects are detected (e.g. person, cell phone)
- 💾 **Save Frames** — capture and save annotated frames as images
- 🖼️ **Saved Frames Gallery** — view all previously saved frames in the app

---

## ⚙️ Installation

Make sure you have **Python 3.10+** installed, then run:

```bash
pip install streamlit streamlit-webrtc ultralytics opencv-python av torch torchvision numpy pillow scipy matplotlib
```

---

## ▶️ How to Run

```bash
streamlit run app.py
```

Then open your browser and go to:
```
http://localhost:8501
```

---

## 🖥️ Expected Output
1. A browser-based app with the title **"Live Object Detection & Tracing"**
2. Webcam feed with bounding boxes and labels on detected objects
3. Live object count displayed on the video and sidebar
4. Alert banner when a selected object is detected
5. Ability to capture and save frames with one click

---

## 📦 Requirements
```
streamlit
streamlit-webrtc
ultralytics
opencv-python
av
numpy
pillow
scipy
matplotlib
```

---

## 🤖 Model Used
- **YOLOv8n** (Nano) — lightweight and fast, trained on the COCO dataset
- Detects **80 object classes** including person, cell phone, bottle, chair, laptop, and more

---

## 👩‍💻 Developed by
**Patricia Martinez**  
BSCS 3 — Dr. Emilio B. Espinosa Sr. Memorial State College of Agriculture and Technology
