import cv2
import numpy as np
import argparse

# Set up the argument parser with flags (e.g., --video, --annotations, --output)
parser = argparse.ArgumentParser(description="Add bounding boxes and track IDs to video.")
parser.add_argument('--v', type=str, required=True, help="Path to the input video file")
parser.add_argument('--a', type=str, required=True, help="Path to the input annotation tracking file")
parser.add_argument('--o', type=str, required=True, help="Path to save the output video with annotations")

# Parse the arguments
args = parser.parse_args()

# Open the video
cap = cv2.VideoCapture(args.v)

# Get video information (frame size, fps, etc.)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare the output video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(args.o, fourcc, fps, (frame_width, frame_height))

# Read tracking annotation file and store the data
tracking_data = {}
with open(args.a, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        frame_id = int(parts[0])
        track_id = int(parts[1])
        bbox_left = float(parts[2])
        bbox_top = float(parts[3])
        bbox_width = float(parts[4])
        bbox_height = float(parts[5])
        
        if frame_id not in tracking_data:
            tracking_data[frame_id] = []
        
        tracking_data[frame_id].append((track_id, bbox_left, bbox_top, bbox_width, bbox_height))

# Process each frame of the video
frame_id = 0
while True:
    ret, frame = cap.read()
    
    if not ret:
        break  # If there are no more frames, exit the loop
    
    # If tracking data exists for this frame, draw the bounding boxes and track IDs
    if frame_id in tracking_data:
        for (track_id, bbox_left, bbox_top, bbox_width, bbox_height) in tracking_data[frame_id]:
            # Draw the bounding box in red
            color = (0, 0, 255)  # Red in BGR format
            cv2.rectangle(frame, (int(bbox_left), int(bbox_top)), 
                          (int(bbox_left + bbox_width), int(bbox_top + bbox_height)), color, 2)
            
            # Draw the track_id above the bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(track_id)
            text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
            text_x = int(bbox_left)
            text_y = int(bbox_top) - 10  # Place text a bit above the box
            cv2.putText(frame, text, (text_x, text_y), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Write the frame with bounding boxes and track IDs to the output video
    out.write(frame)
    
    frame_id += 1

# Release resources
cap.release()
out.release()

print(f"Annotated video with bounding boxes and track IDs saved to {args.o}")
