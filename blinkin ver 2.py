import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np
import time

def euclidean_distance(ptA, ptB):
    return dist.euclidean(ptA, ptB)

def eye_aspect_ratio(eye):
    A = euclidean_distance(eye[1], eye[5])
    B = euclidean_distance(eye[2], eye[4])
    C = euclidean_distance(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def calculate_dynamic_threshold(ear_history, default_threshold=0.25, sensitivity=0.05):
    if not ear_history:
        return default_threshold
    mean_ear = np.mean(ear_history)
    return max(mean_ear - sensitivity, 0.2)  # Adjust sensitivity as needed, ensure it doesn't go below a sensible lower bound

# Constants
BLINK_CONSEC_FRAMES = 3
SHORT_BLINK_DURATION = 0.3
LONG_BLINK_DURATION = 0.5
DEBOUNCE_TIME = 1.0  # Time in seconds to ignore blinks after one is detected
EAR_HISTORY_LENGTH = 30

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture(0)

frame_counter = 0
blink_counter = 0
long_blink_counter = 0
short_blink_counter = 0
blink_start_time = None
last_blink_time = None
ear_history = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[l_start:l_end]
        right_eye = shape[r_start:r_end]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if len(ear_history) >= EAR_HISTORY_LENGTH:
            ear_history.pop(0)
        ear_history.append(ear)

        dynamic_threshold = calculate_dynamic_threshold(ear_history)
        
        if ear < dynamic_threshold:
            current_time = time.time()
            if blink_start_time is None:
                blink_start_time = current_time
            
            if last_blink_time is None or (current_time - last_blink_time) > DEBOUNCE_TIME:
                frame_counter += 1
            
            if frame_counter >= BLINK_CONSEC_FRAMES:
                blink_duration = current_time - blink_start_time
                if blink_duration >= LONG_BLINK_DURATION:
                    long_blink_counter += 1
                elif blink_duration >= SHORT_BLINK_DURATION:
                    short_blink_counter += 1
                blink_counter += 1
                print(f"Blink detected! Total: {blink_counter}, Long: {long_blink_counter}, Short: {short_blink_counter}")
                last_blink_time = current_time
                frame_counter = 0
        else:
            blink_start_time = None
            frame_counter = 0
        
        # Pupil tracking and gaze estimation removed for simplicity; add if needed.

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
