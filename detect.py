import os
import cv2
import time
import shutil
import mediapipe as mp
import numpy as np
from datetime import datetime
from ultralytics import YOLO

screenshot_dir = 'captured_images'

def clear_images_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

# Clear captured_images directory at the start
clear_images_directory(screenshot_dir)

# Load YOLO model
model_path = 'C:/Users/andyma/Documents/vrchat_pose_detection/train_datasets/yolov8n.pt'
model = YOLO(model_path)

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

processed_images = set()
last_movement_time = datetime.now()

def is_waving(landmarks):
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    return (left_elbow.y < left_shoulder.y) and (left_wrist.y < left_elbow.y)

def draw_text_with_background(image, text, position, font, scale, text_color, bg_color, thickness):
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(image, (x, y - text_h - baseline), (x + text_w, y + baseline), bg_color, -1)
    cv2.putText(image, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)

def monitor_images(screenshot_dir):
    global last_movement_time
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        cv2.namedWindow('Mediapipe Feed', cv2.WINDOW_NORMAL)  # Create one window
        
        try:
            while True:
                if not os.path.exists(screenshot_dir):
                    clear_images_directory(screenshot_dir)
                
                for image_name in os.listdir(screenshot_dir):
                    image_path = os.path.join(screenshot_dir, image_name)
                    
                    if image_path in processed_images:
                        continue
                    
                    frame = cv2.imread(image_path)
                    
                    if frame is None:
                        continue

                    processed_images.add(image_path)

                   # Perform YOLO inference
                    yolo_results = model(frame)[0]
                    detections = yolo_results.boxes  # Accessing boxes from YOLOv8 results
                    
                    # Print detections
                    print("YOLO Detections:")
                    for box in detections:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        label = yolo_results.names[cls]
                        print(f"Box coordinates: ({x1}, {y1}), ({x2}, {y2}) - Confidence: {conf} - Class: {label}")

                    
                    # Recolor image to RGB
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False
                    
                    # Make Mediapipe detection
                    results = pose.process(image_rgb)
                    
                    # Recolor back to BGR
                    image_rgb.flags.writeable = True
                    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Extract landmarks
                    try:
                        landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
                        if landmarks:
                            if is_waving(landmarks):
                                draw_text_with_background(image, 'Wave Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 128, 255), 2)
                                print("Wave Detected!")
                                last_movement_time = datetime.now()
                            
                            if (datetime.now() - last_movement_time).seconds > 10:
                                draw_text_with_background(image, 'No Movement Detected!', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 255), 2)
                                print("No Movement Detected!")
                        
                        if results.pose_landmarks:
                            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                        
                        # Render YOLO detections 
                        for box in detections:
                            x1, y1, x2, y2 = box.xyxy[0]
                            conf = box.conf[0]
                            cls = int(box.cls[0])
                            label = yolo_results.names[cls]
                            
                            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(image, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    except Exception as e:
                        print(f"Error processing image: {e}")

                    cv2.imshow('Mediapipe Feed', image)
    
                    key = cv2.waitKey(10)
                    if key == ord('q') or key == 27:
                        raise KeyboardInterrupt  
    
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("Exiting program.")
        
        finally:
            cv2.destroyAllWindows()
            clear_images_directory(screenshot_dir)
