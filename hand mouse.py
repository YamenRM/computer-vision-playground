import cv2
import mediapipe as mp
import pyautogui 
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
        frame = cv2.resize(frame, (1366, 768))


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

# controling the mouse with the tip of the index finger and the thumb

                # Get the dimensions of the frame
                height, width, _ = frame.shape
                # Extract the x and y coordinates of the hand landmarks
                x_coordinates = [landmark.x for landmark in hand_landmarks]
                y_coordinates = [landmark.y for landmark in hand_landmarks]

                index_tip = (int(x_coordinates[8] * width), int(y_coordinates[8] * height))


                # drawing a ciurle on the tip of the index finger
                cv2.circle(frame, (index_tip), 10, (255, 255, 255), -1)
                
            

                #Move the mouse cursor to the position of the index finger tip intstantly
                pyautogui.moveTo((index_tip))
                

                # click the mouse when the distance between the index finger tip and thumb tip is less than a certain threshold
                distance = ((x_coordinates[8] - x_coordinates[12]) ** 2 + (y_coordinates[8] - y_coordinates[12]) ** 2) ** 0.5
                if distance < 0.08 and distance > 0.06:
                   pyautogui.click()
                   delay = 0.2
                   time.sleep(delay)  

                   # add a holding action when the distance is less than a smaller threshold
                elif distance < 0.06:
                    pyautogui.mouseDown()
                    # add a release action when the distance is greater than the smaller threshold
                elif distance > 0.06:
                    pyautogui.mouseUp()



        cv2.imshow('mouse controler', frame)
        cv2.resizeWindow('mouse controler', 1366, 768)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
