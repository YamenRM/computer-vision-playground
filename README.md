# Computer Vision Playground

Computer Vision Playground is an interactive computer vision application built with Streamlit, MediaPipe, OpenCV, and PyAutoGUI.  
The project demonstrates real-time face tracking, hand tracking, gesture control, and visual effects using a webcam.

---

## Features

### Face Landmark Detection
Detects and tracks facial landmarks in real time using your webcam.  
The application visualizes facial key points to demonstrate how computer vision models understand facial structure.

### Hand Landmark Detection
Tracks hand movements and identifies finger positions using MediaPipe's hand tracking model.

### Gesture-Based Mouse Control
Hand gestures are translated into mouse actions such as:
- Cursor movement
- Clicking
- Basic interaction without touching the mouse

### AR Hand Effect
A transparent **Doctor Strange–style hand effect** is overlaid on the detected hand.  
The image follows the hand in real time, creating a simple augmented reality experience.

---

## Tech Stack

- Python
- Streamlit
- MediaPipe
- OpenCV
- PyAutoGUI

---

## Assets and Models

The application uses external model and image files.

- `models/` → [handlandmarker model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) ,
  [facelandmarker model](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)
- `assets/` → [dr.strange hand img](https://pngtree.com/so/dr-strange)

---

## How It Works

1. The webcam captures video frames.
2. OpenCV processes the frames.
3. MediaPipe detects face and hand landmarks.
4. The program interprets gestures or positions.
5. Visual effects or mouse actions are applied in real time.

---
## How to Use

Follow the instructions below to work with **Computer Vision Playground**.

### 1- Installation

```bash
git clone https://github.com/yourusername/computer-vision-playground.git
cd computer-vision-playground
pip install -r requirements.txt
```
### 2- assgine the model paths in the three files and the img path on the hand_landmark_mask.py

### 3- change the parmeters(optinal)

feel free to change the options of the models and other changeable parameters.

### 4- run main.py

```bash
streamlit run main.py
```
### 5- Have fun


## Author
YamenRM | 2026 
