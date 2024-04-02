import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import time

# Function to calculate the Euclidean distance between two points
def euclidean_distance(ptA, ptB):
    return dist.euclidean(ptA, ptB)

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = euclidean_distance(eye[1], eye[5])
    B = euclidean_distance(eye[2], eye[4])
    
    # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = euclidean_distance(eye[0], eye[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    return ear

# Constants
EAR_THRESHOLD = 0.25
BLINK_CONSEC_FRAMES = 3
SHORT_BLINK_DURATION = 0.3  # Adjust as needed
LONG_BLINK_DURATION = 0.5   # Adjust as needed

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("downloads/shape_predictor_68_face_landmarks.dat")

# Get the indices for the left and right eyes using face_utils
(l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize frame counter and blink counter
frame_counter = 0
blink_counter = 0
long_blink_counter = 0
short_blink_counter = 0
blink_start_time = None

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = detector(gray, 0)
    
    for face in faces:
        # Detect facial landmarks
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        # Extract left and right eye coordinates
        left_eye = shape[l_start:l_end]
        right_eye = shape[r_start:r_end]
        
        # Calculate eye aspect ratios
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        # Calculate the average eye aspect ratio
        ear = (left_ear + right_ear) / 2.0
        
        # Check if the eye aspect ratio is below the threshold
        if ear < EAR_THRESHOLD:
            if blink_start_time is None:
                blink_start_time = time.time()
            frame_counter += 1
            
            # If eyes closed for a sufficient number of frames, increment blink counter
            if frame_counter >= BLINK_CONSEC_FRAMES:
                blink_end_time = time.time()
                blink_duration = blink_end_time - blink_start_time
                if blink_duration >= LONG_BLINK_DURATION:
                    long_blink_counter += 1
                    print("Long blink detected! Total long blinks:", long_blink_counter)
                elif blink_duration >= SHORT_BLINK_DURATION:
                    short_blink_counter += 1
                    print("Short blink detected! Total short blinks:", short_blink_counter)
                blink_counter += 1
                print("Blink detected! Total blinks:", blink_counter)
                frame_counter = 0
        else:
            blink_start_time = None
            frame_counter = 0
            
        # Pupil tracking
        # Assume the center of the eye region is the approximate location of the pupil
        left_eye_center = left_eye.mean(axis=0).astype("int")
        right_eye_center = right_eye.mean(axis=0).astype("int")
        
        # Draw circles around the estimated pupil locations
        cv2.circle(frame, tuple(left_eye_center), 2, (0, 255, 0), -1)
        cv2.circle(frame, tuple(right_eye_center), 2, (0, 255, 0), -1)
        
        # Calculate direction of gaze based on relative positions of pupils
        if left_eye_center[1] > right_eye_center[1]:
            direction = "Up"
        elif left_eye_center[1] < right_eye_center[1]:
            direction = "Down"
        else:
            direction = "Straight"
        
        if left_eye_center[0] > right_eye_center[0]:
            direction += ", Right"
        elif left_eye_center[0] < right_eye_center[0]:
            direction += ", Left"
        
        # Display the direction of gaze
        cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow("Frame", frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()