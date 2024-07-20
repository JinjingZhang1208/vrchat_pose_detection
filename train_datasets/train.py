import os
from IPython import display
from ultralytics import YOLO
from roboflow import Roboflow

# Specify the correct path to the YOLOv8 model weights
model_path = 'C:/Users/andyma/Documents/vrchat_pose_detection/train_datasets/yolov8n.pt'

# Load the YOLO model
model = YOLO(model_path)

# Initialize Roboflow
rf = Roboflow(api_key="eW43mQoE4D7QBLaccpat")
project = rf.workspace("annazhang1208").project("vr_pose")
version = project.version(1)
dataset = version.download("yolov5")

# Correct dataset path
data_path = 'C:/Users/andyma/Documents/vrchat_pose_detection/train_datasets/vr_pose-1/data.yaml'

# Verify dataset location
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

# Define the training task
task = {
    'mode': 'train',
    'model': model_path,
    'data': data_path,
    'epochs': 25,
    'imgsz': 800,
    'plots': True
}

# Execute the training task
model.train(data=task['data'], epochs=task['epochs'], imgsz=task['imgsz'], plots=task['plots'])
