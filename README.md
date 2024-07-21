# vrchat_pose_detection

## Goal
The primary objective of this project is to detect avatars' waving and identify periods of no movement in the previous 10 seconds.

## Overview
- Utilize YOLOv8 to detect objects as bounding boxes.
- Use MediaPipe to analyze the motions/poses within the detected boxes.
- Annotate custom collected datasets using Roboflow.
- Train datasets using PyTorch.

## Dataset
rf = Roboflow(api_key="eW43mQoE4D7QBLaccpat")
project = rf.workspace("annazhang1208").project("vr_pose_avatars")
version = project.version(1)
dataset = version.download("yolov5")

