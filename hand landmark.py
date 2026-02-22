import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles


# initialize mediapipe Hand landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker=vision.HandLandmarker

# drawing parameters
MARGIN = 10  
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)




# result callback function to store the latest hand landmarker result  
latest_result = None

def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# setup mediapipe hand landmarker options
base_options = BaseOptions (model_asset_path='E:/yamen models/mediapipe/hand_landmarker.task')
options = HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,  
    min_hand_detection_confidence=0.5, 
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback
)

# start the live cam and hand detection loop
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1) 
        frame = cv2.resize(frame, (1280, 720))


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



        cv2.imshow('hand landmark', frame)
        cv2.resizeWindow('hand landmark', 1280, 720)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
