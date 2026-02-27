import cv2
import mediapipe as mp
import pyautogui 
import streamlit as st
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

def hand_mouse(placeholder):

    # initialize mediapipe Hand landmarker
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
    VisionRunningMode = mp.tasks.vision.RunningMode
    HandLandmarker=vision.HandLandmarker



    # result callback function to store the latest hand landmarker result  
    latest_result = None

    def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        nonlocal latest_result
        latest_result = result

    # setup mediapipe hand landmarker options
    base_options = BaseOptions (model_asset_path='models/hand_landmarker.task')
    options = HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,  
        min_hand_detection_confidence=0.7, 
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.9,
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=result_callback
    )

    # start the live cam and hand detection loop
    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


        
        while cap.isOpened() and st.session_state.get('run_mouse', False):
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1) 



            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # use the current timestamp in milliseconds for the face landmarker
            timestamp = int(time.time_ns() // 1000000)  
            landmarker.detect_async(mp_image, timestamp)

            # draw the hand landmarks on the frame
            if latest_result and latest_result.hand_landmarks:
                for hand_landmarks in latest_result.hand_landmarks:
                    drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=drawing_styles.get_default_hand_connections_style())

                # hand control logic
                    height, width, _ = frame.shape


                    x_coordinates = [landmark.x for landmark in hand_landmarks]
                    y_coordinates = [landmark.y for landmark in hand_landmarks]


                    # get the centre of the tringle formed by the wrist, index finger mcp and pinky mcp to use it as a reference point for the mouse cursor
                    wrist_tip = (int(x_coordinates[0] * width), int(y_coordinates[0] * height))
                    index_mcp = (int(x_coordinates[5] * width), int(y_coordinates[5] * height))
                    picky_mcp = (int(x_coordinates[17] * width), int(y_coordinates[17] * height))

                    middle_point = (int((wrist_tip[0] + index_mcp[0] + picky_mcp[0]) / 3), int((wrist_tip[1] + index_mcp[1] + picky_mcp[1]) / 3))

                    cv2.circle(frame, (middle_point), 10, (0, 255, 255), -1)

                    pyautogui.moveTo((middle_point))
                    pyautogui.FAILSAFE = False
                    pyautogui.PAUSE = 0

                    # makeing the mouse functions
                    index_tip = (int(x_coordinates[8] * width), int(y_coordinates[8] * height))
                    thumb_tip = (int(x_coordinates[4] * width), int(y_coordinates[4] * height))
                    middle_finger_tip = (int(x_coordinates[12] * width), int(y_coordinates[12] * height))
                    ring_tip = (int(x_coordinates[16] * width), int(y_coordinates[16] * height))
                    pinky_tip = (int(x_coordinates[20] * width), int(y_coordinates[20] * height))
                     
                    



                    click_distance = ((index_tip[0] - thumb_tip[0]) ** 2 + (index_tip[1] - thumb_tip[1]) ** 2) ** 0.5
                    right_click_distance = ((middle_finger_tip[0] - thumb_tip[0]) ** 2 + (middle_finger_tip[1] - thumb_tip[1]) ** 2) ** 0.5
                    drag_distance = ((ring_tip[0] - thumb_tip[0]) ** 2 + (ring_tip[1] - thumb_tip[1]) ** 2) ** 0.5
                    scroll_distance = ((pinky_tip[0] - thumb_tip[0]) ** 2 + (pinky_tip[1] - thumb_tip[1]) ** 2) ** 0.5

                    
                    if click_distance < 40:  
                        pyautogui.click() 
                        cv2.putText(frame, 'Click', (0 ,height), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)   
                    elif drag_distance < 40:
                        pyautogui.mouseDown()
                        cv2.putText(frame, 'hold', (0 ,height), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                    elif right_click_distance < 40:
                        pyautogui.click(button='right') 
                        cv2.putText(frame, 'right-click', (0 ,height), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                    elif scroll_distance < 50 and scroll_distance > 0:
                        pyautogui.scroll(50)
                        cv2.putText(frame, 'scroll-up', (0 ,height), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                    elif scroll_distance > 60 and scroll_distance < 90:
                        pyautogui.scroll(-50)
                        cv2.putText(frame, 'scroll-down', (0 ,height), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                    else:
                        pyautogui.moveTo((middle_point))


            placeholder.image(frame, channels='BGR' , width='stretch' ) 

    cap.release()
    cv2.destroyAllWindows()


