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

## How It Works

1. The webcam captures video frames.
2. OpenCV processes the frames.
3. MediaPipe detects face and hand landmarks.
4. The program interprets gestures or positions.
5. Visual effects or mouse actions are applied in real time.

---

## Installation

```bash
git clone https://github.com/yourusername/computer-vision-playground.git
cd computer-vision-playground
pip install -r requirements.txt
