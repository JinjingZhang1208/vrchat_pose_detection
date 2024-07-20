import cv2
import mediapipe as mp
import numpy as np
import os
import time
import shutil
from datetime import datetime

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Directory containing screenshots
screenshot_dir = 'captured_images'

# Function to clear captured_images directory
def clear_images_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

# Clear captured_images directory at the start
clear_images_directory(screenshot_dir)

processed_images = set()

def is_waving(landmarks):
    if landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y < landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y:
        if landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y < landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y:
            return True
    return False

def calculate_movement(kp1, kp2):
    speed = np.linalg.norm(np.array(kp2) - np.array(kp1))
    direction = np.array(kp2) - np.array(kp1)
    return speed, direction

def is_moving(prev_landmarks, curr_landmarks, threshold=0.02):
    if prev_landmarks is None or curr_landmarks is None:
        return True  # Consider initial state as moving
    
    # Calculate the distance moved by hips
    prev_left_hip = prev_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    curr_left_hip = curr_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    
    distance = np.sqrt((curr_left_hip.x - prev_left_hip.x) ** 2 + (curr_left_hip.y - prev_left_hip.y) ** 2)
    
    return distance > threshold

def draw_text_with_background(image, text, position, font, scale, text_color, bg_color, thickness):
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(image, (x, y - text_h - baseline), (x + text_w, y + baseline), bg_color, -1)
    cv2.putText(image, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)

def monitor_images(screenshot_dir):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        prev_landmarks = None
        wave_detected = False
        
        cv2.namedWindow('Mediapipe Feed', cv2.WINDOW_NORMAL)  # Create one window
        
        try:
            while True:
                # Check if directory exists and capture frames
                if not os.path.exists(screenshot_dir):
                    clear_images_directory(screenshot_dir)
                
                for image_name in os.listdir(screenshot_dir):
                    image_path = os.path.join(screenshot_dir, image_name)
                    
                    if image_path in processed_images:
                        continue
                    
                    # Read the image
                    frame = cv2.imread(image_path)
                    
                    if frame is None:
                        continue
    
                    processed_images.add(image_path)
    
                    # Recolor image to RGB
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    
                    # Make detection
                    results = pose.process(image)
                
                    # Recolor back to BGR
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # Extract landmarks
                    try:
                        landmarks = results.pose_landmarks.landmark
                        
                        # Check for waving gesture
                        if landmarks and is_waving(landmarks):
                            draw_text_with_background(image, 'Wave Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 128, 255), 2)
                            print("Wave Detected!")
                            last_movement_time = datetime.now()
                            wave_detected = True
                        
                        # Check if there is no movement in 10 seconds
                        if (datetime.now() - last_movement_time).seconds > 10:
                            draw_text_with_background(image, 'No Movement Detected!', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 255), 2)
                            print("No Movement Detected!")
                        
                        prev_landmarks = landmarks
                    
                    except Exception as e:
            
                        pass
                    
                    # Render detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                    
                    cv2.imshow('Mediapipe Feed', image)
    
                    key = cv2.waitKey(10)
                    if key == ord('q') or key == 27:  # 'q' key or Esc key
                        raise KeyboardInterrupt  
    
                time.sleep(1)  # Adjust sleep time as necessary to control the checking frequency
        
        except KeyboardInterrupt:
            print("Exiting program.")
        
        finally:
            cv2.destroyAllWindows()
            clear_images_directory(screenshot_dir)