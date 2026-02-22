import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles


# initialize mediapipe face landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker=vision.FaceLandmarker


# result callback function to store the latest hand landmarker result  
latest_result = None

def result_callback(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# setup mediapipe hand landmarker options
base_options = BaseOptions (model_asset_path='E:/yamen models/mediapipe/face_landmarker.task')
options = FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=2,  
    min_face_detection_confidence=0.5, 
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback
)

# start the live cam and hand detection loop
with FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1) 
        frame = cv2.resize(frame, (1280, 600))


        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # use the current timestamp in milliseconds for the face landmarker
        timestamp = int(time.time_ns() // 1000000)  
        landmarker.detect_async(mp_image, timestamp)

        # draw the face landmarks on the frame
        if latest_result and latest_result.face_landmarks:
            for face_landmarks in latest_result.face_landmarks:
                # style 1
                drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
                # style 2
                drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())


        cv2.imshow('testing', frame)
        cv2.resizeWindow('testing', 1280, 720)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
