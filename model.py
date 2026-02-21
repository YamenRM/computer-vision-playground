import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# initialize mediapipe hand landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# result callback function to store the latest hand landmarker result  
latest_result = None

def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# setup mediapipe hand landmarker options
base_options = BaseOptions (model_asset_path='E:/yamen models/mediapipe/hand_landmarker.task')
options = HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,  
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
        frame = cv2.resize(frame, (1280, 600))


        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # use the current timestamp in milliseconds for the hand landmarker
        timestamp = int(time.time_ns() // 1000000)  
        landmarker.detect_async(mp_image, timestamp)

        # draw the hand landmarks and connections on the frame
        if latest_result is not None and latest_result.hand_landmarks:
            h, w, _ = frame.shape
            
            # list of connections between hand landmarks based on mediapipe hand landmark model (from palm to fingertips)
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),           # الإبهام
                (0, 5), (5, 6), (6, 7), (7, 8),           # السبابة
                (0, 9), (9, 10), (10, 11), (11, 12),     # الوسطى
                (0, 13), (13, 14), (14, 15), (15, 16),   # البنصر
                (0, 17), (17, 18), (18, 19), (19, 20)    # الخنصر
            ]
            
            for hand_landmarks in latest_result.hand_landmarks:
                # draw the lines 
                for connection in connections:
                    start_idx, end_idx = connection
                    start_landmark = hand_landmarks[start_idx]
                    end_landmark = hand_landmarks[end_idx]
                    
                    x1 = int(start_landmark.x * w)
                    y1 = int(start_landmark.y * h)
                    x2 = int(end_landmark.x * w)
                    y2 = int(end_landmark.y * h)
                    
                    cv2.line(frame, (x1, y1 ), (x2, y2 ), (100, 150, 255), 2)
                
                # draw the circles on the landmarks
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)



        cv2.imshow('testing', frame)
        cv2.resizeWindow('testing', 1280, 720)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
