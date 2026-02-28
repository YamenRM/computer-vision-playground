import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles


draw_color = (0 , 255 , 255)

erase_color = (0 , 0 , 0)

if 'canvas' not in st.session_state:
    st.session_state.canvas = np.zeros((480, 640, 3), dtype=np.uint8)

def drawing_mode(placeholder):
    
    # initialize mediapipe Hand landmarker
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
    VisionRunningMode = mp.tasks.vision.RunningMode
    HandLandmarker=vision.HandLandmarker

  
    latest_result = None
    last_x, last_y = 0, 0

    def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        nonlocal latest_result
        latest_result = result

    # setup mediapipe hand landmarker options
    base_options = BaseOptions (model_asset_path='models/hand_landmarker.task')
    options = HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,  
        min_hand_detection_confidence=0.7, 
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=result_callback
    )



    # start the live cam and hand detection loop
    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

       

        while cap.isOpened() and st.session_state.get('run_drawing_mode', False):
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480)) 
           

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            
            
            timestamp = int(time.time_ns() // 1000000)  
            landmarker.detect_async(mp_image, timestamp)

            # drawing logic
            if latest_result and latest_result.hand_landmarks:
                for hand_landmarks in latest_result.hand_landmarks:
                
                    index_tip = hand_landmarks[8]
                    index_mcp = hand_landmarks[6]
                    middle_tip = hand_landmarks[12]
                    middle_mcp = hand_landmarks[10]
                    ring_tip = hand_landmarks[16]
                    ring_mcp = hand_landmarks[14]


                    curr_x, curr_y = int(index_tip.x * 640), int(index_tip.y * 480)
            

                    if  ring_tip.y > ring_mcp.y  and index_tip.y > index_mcp.y and middle_tip.y > middle_mcp.y:
                #  NEUTRAL MODE 
                        cv2.putText(frame, "NEUTRAL", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        last_x, last_y = 0, 0 # Reset tracking so it doesn't jump when you reopen
                    
                    elif ring_tip.y < ring_mcp.y  and index_tip.y < index_mcp.y and middle_tip.y < middle_mcp.y:
                        st.session_state.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                        last_x, last_y = 0, 0
                        cv2.putText(frame, "CLEAR", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                    else:
                # 2. Hand is open, now decide between DRAW and ERASE
                        dist = ((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)**0.5

                        if dist < 0.08:
                    # --- ERASING MODE ---
                            cv2.circle(frame, (curr_x, curr_y), 40, (255, 255, 255), 2)
                            cv2.putText(frame, "ERASING", (curr_x + 35, curr_y + 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.circle(st.session_state.canvas, (curr_x, curr_y), 40, erase_color, -1)
                            last_x, last_y = 0, 0 
                        else: 
                    # --- DRAWING MODE ---
                            cv2.circle(frame, (curr_x, curr_y), 8, draw_color, -1)
                            cv2.putText(frame, "DRAWING", (curr_x + 15, curr_y + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1)

                            if last_x != 0 and last_y != 0:
                                cv2.line(st.session_state.canvas, (last_x, last_y), (curr_x, curr_y), draw_color, 5)
                            last_x, last_y = curr_x, curr_y

            else:
                last_x, last_y = 0, 0
            
            canvas_gray = cv2.cvtColor(st.session_state.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)


            img_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            img_fg = cv2.bitwise_and(st.session_state.canvas, st.session_state.canvas, mask=mask)
            combined_frame = cv2.add(img_bg, img_fg)
                

                            
            placeholder.image(combined_frame, channels='BGR')

    cap.release()
    cv2.destroyAllWindows()
