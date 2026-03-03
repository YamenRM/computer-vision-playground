# Computer Vision Experiments Lab

Interactive real-time computer vision experiments built with Python.
The application demonstrates face tracking, hand tracking, gesture-based control, augmented reality overlays, and air drawing using a webcam.

---

# Features

## Face Landmark Detection

Real-time facial landmark tracking using MediaPipe.
The system detects key facial points and visualizes them directly on the video stream.

## Hand Landmark Detection

Tracks finger positions and hand structure with high accuracy.

## AR Hand Effect

A transparent Doctor Strange–style visual effect follows the detected hand in real time, creating a simple augmented reality interaction.

## Gesture-Based Mouse Control

Control your computer using hand gestures:

* Move cursor
* Click actions
* Touchless interaction

## Virtual Air Drawing

Turn your index finger into a digital brush:

* Draw in real time
* Clear the canvas
* Smooth motion tracking

---

# Tech Stack

* Python
* Streamlit
* MediaPipe
* OpenCV
* PyAutoGUI

---

# Project Structure

```
Computer-Vision-Experiments-Lab
│
├── src
│   ├── main.py
│   ├── facelandmark.py
│   ├── hand_landmark_mask.py
│   ├── mouse_control.py
│   └── drawing_mode.py
│
├── models
│   ├── hand_landmarker.task
│   └── face_landmarker.task
│
├── assets
│   ├── dr_strange_hand.png
│   └── demo.gif
│
├── requirements.txt
└── README.md
```

---

# Installation

```bash
git clone https://github.com/yourusername/Computer-Vision-Experiments-Lab.git
cd Computer-Vision-Experiments-Lab
pip install -r requirements.txt
```

---

# Run the Application

```bash
streamlit run src/main.py
```

Open the local Streamlit address shown in the terminal.

Allow camera access when prompted.

---

# Models

Used Models

[Hand Landmarker](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)

[Face Landmarker](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)

---

# How to Use

1. Launch the application.
2. Select a mode from the interface.
3. Position yourself in front of the webcam.
4. Try different gestures.

Available modes:

* Face Landmark Detection
* Hand Landmark Mask
* Hand Mouse Control
* Drawing Mode

---

# Future Improvements

* Gesture shortcuts
* Multiple brush colors
* Adjustable brush size
* Performance optimization
* Additional AR effects

---

# Author

YamenRM
AI Engineering Student
