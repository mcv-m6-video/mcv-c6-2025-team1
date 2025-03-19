import argparse
import os
import time
import cv2
from ultralytics import YOLO  # Ensure you have YOLO installed

# Argument parser setup
parser = argparse.ArgumentParser(description="Run YOLO inference on a video.")
parser.add_argument("--model", type=str, required=True, help="Path to the YOLO model.")
parser.add_argument("--input", type=str, required=True, help="Path to the input video.")
parser.add_argument("--output", type=str, required=True, help="Path to save the output video.")
args = parser.parse_args()

# Load the YOLO model
model = YOLO(args.model)

# Open the video file
cap = cv2.VideoCapture(args.input)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create VideoWriter object 
out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Process video frame by frame
start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video
    
    # Perform YOLO inference on the frame
    results = model.predict(source=frame, classes=[2])  # Detect objects of class 2
    
    # Draw detections on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = f"{box.cls[0].item():.0f}: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 7)
    
    # Write the frame to the output video
    out.write(frame)

# Release everything
cap.release()
out.release()

# Compute total processing time
inference_time = time.time() - start_time
print(f"Processing completed in {inference_time:.2f} seconds.")