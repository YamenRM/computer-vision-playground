import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import time
import random
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles


# overlay func for transparent images
def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w = background.shape[:2]
    h, w = overlay.shape[:2]
    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0: return background
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, bg_w), min(y + h, bg_h)
    overlay_x1, overlay_y1 = x1 - x, y1 - y
    overlay_x2, overlay_y2 = overlay_x1 + (x2 - x1), overlay_y1 + (y2 - y1)
    if overlay.shape[2] == 4: 
        overlay_image = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, :3]
        mask = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
        for c in range(3):
            background[y1:y2, x1:x2, c] = (mask * overlay_image[:, :, c] + (1.0 - mask) * background[y1:y2, x1:x2, c])
    else:
        background[y1:y2, x1:x2] = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    return background

# helper function to determine if hand is open based on landmark positions
def is_hand_open(hand_landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    open_fingers = 0
    for t, p in zip(tips, pips):
        if hand_landmarks[t].y < hand_landmarks[p].y: open_fingers += 1
    return open_fingers >= 4

def hand_landmark_mask(placeholder):
    
    # initialize mediapipe Hand landmarker
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
    VisionRunningMode = mp.tasks.vision.RunningMode
    HandLandmarker=vision.HandLandmarker

    angle = 0 
    latest_result = None

    def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        nonlocal latest_result
        latest_result = result

    # setup mediapipe hand landmarker options
    base_options = BaseOptions (model_asset_path='models/hand_landmarker.task')
    options = HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,  
        min_hand_detection_confidence=0.5, 
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=result_callback
    )


    overlay_img = cv2.imread(r'misc/imege.png', cv2.IMREAD_UNCHANGED)

    # start the live cam and hand detection loop
    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)



        
        while cap.isOpened() and st.session_state.get('run_hand_landmark_mask', False):
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1) 

            h_frame, w_frame, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            
            
            timestamp = int(time.time_ns() // 1000000)  
            landmarker.detect_async(mp_image, timestamp)

            # draw the custom hand landmarks on the frame
            if latest_result and latest_result.hand_landmarks:
                for hand_landmarks in latest_result.hand_landmarks:
                    if is_hand_open(hand_landmarks) and overlay_img is not None:

                        cx, cy = int(hand_landmarks[9].x * w_frame), int(hand_landmarks[9].y * h_frame)
                    
                        dist = np.linalg.norm(np.array([hand_landmarks[0].x, hand_landmarks[0].y]) - np.array([hand_landmarks[9].x, hand_landmarks[9].y]))
                        size = int(dist * w_frame * 1.8) 
                        
                        if size > 10:
                           
                            angle = (angle + 5) % 360 
                            M = cv2.getRotationMatrix2D((overlay_img.shape[1]//2, overlay_img.shape[0]//2), angle, 1.0)
                            rotated = cv2.warpAffine(overlay_img, M, (overlay_img.shape[1], overlay_img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
                          
                            resized = cv2.resize(rotated, (size, size))

                            glow_size = 5 + int(5 * np.sin(time.time() * 5)) 
                            final_magic = cv2.GaussianBlur(resized, (0,1), glow_size)
                            final_magic = cv2.addWeighted(resized, 0.5, final_magic, 1.5, 0)


                            frame = overlay_transparent(frame, final_magic, cx - size//2, cy - size//2)

                            
            placeholder.image(frame, channels='BGR')

    cap.release()
    cv2.destroyAllWindows()
