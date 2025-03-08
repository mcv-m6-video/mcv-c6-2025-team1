from utils import iou
from collections import defaultdict
import cv2
import numpy as np
from sort import Sort
import ptlflow
from ptlflow.utils.io_adapter import IOAdapter

def compute_optical_flow(img1, img2):
    """
    Compute optical flow using RAFT (Recurrent All-Pairs Field Transforms) between two images.

    Args:
        img1 (np.array): The first image (prev_frame).
        img2 (np.array): The second image (curr_frame).

    Returns:
        np.array: Optical flow in the form of (u, v) components stacked together.
    """
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    model_ptlflow = ptlflow.get_model('rpknet', ckpt_path='things')
    io_adapter = IOAdapter(model_ptlflow, img1_rgb.shape[:2])
    inputs = io_adapter.prepare_inputs([img1_rgb, img2_rgb])

    predictions = model_ptlflow(inputs)
    flow = predictions['flows'][0, 0]  # Remove batch and sequence dimensions
    flow = flow.permute(1, 2, 0)  # Convert from CHW to HWC format

    u = flow[:, :, 0].detach().numpy()  # Horizontal flow component
    v = flow[:, :, 1].detach().numpy()  # Vertical flow component

    return np.dstack((u, v))


def read_detections_from_txt(file_path):
    detections = {}
    detections_vect = []
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split(',')
            frame_id = int(data[0])
            bbox_left = float(data[2])
            bbox_top = float(data[3])
            bbox_width = float(data[4])
            bbox_height = float(data[5])
            confidence_score = float(data[6])

            # Store in dictionary with frame_id as key
            if frame_id not in detections:
                detections[frame_id] = []
            
            detections[frame_id].append({
                "bb_left": bbox_left,
                "bb_top": bbox_top,
                "bb_w": bbox_width,
                "bb_h": bbox_height,
                "score": confidence_score
            })

            detections_vect.append([frame_id, bbox_left, bbox_top,bbox_width,bbox_height, confidence_score])

    return detections, detections_vect


def write_results_to_txt(track_eval_format, output_path):
    with open(output_path, 'w') as file:
        file.writelines(track_eval_format)
    print(f"Results written to {output_path}")





detections, detections_vect = read_detections_from_txt('/ghome/c5mcv01/mcv-c6-2025-team1/week3/src/tracking/detections_yolo.txt')  
#print(f"Detections: {detections}")

# Create instance of the SORT tracker (default params: max_age=1, min_hits=3, iou_threshold=0.3)
mot_tracker = Sort(max_age = 21, min_hits=2, iou_threshold=0.2) 

# Open the video
cap = cv2.VideoCapture("/ghome/c5mcv01/mcv-c6-2025-team1/data/AICity_data/train/S03/c010/vdo.avi")

# Get video information (frame size, fps, etc.)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare the output video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("/ghome/c5mcv01/mcv-c6-2025-team1/week3/src/tracking/kf.avi", fourcc, fps, (frame_width, frame_height))

# Initialize variables for optical flow
prev_frame = None
prev_gray = None
prev_points = None

# Process each frame of the video
track_eval_format = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for optical flow calculation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get the current frame number
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    print("-" * 50)
    print(f"Processing frame {frame_number}")
    
    actual_bb = [detection[1:5] for detection in detections_vect if detection[0] == frame_number]  
    actual_bb = np.array(actual_bb)

    if prev_frame is not None:
        # Use optical flow to track motion between frames
        flow = compute_optical_flow(prev_frame, frame)
        
        # Use the optical flow to adjust bounding box positions
        for i, bbox in enumerate(actual_bb):
            x1, y1, w, h = bbox
            flow_x = flow[int(y1), int(x1)][0]  # Horizontal flow component at top-left corner
            flow_y = flow[int(y1), int(x1)][1]  # Vertical flow component at top-left corner
            actual_bb[i, 0] += flow_x  # Update x-coordinate based on flow
            actual_bb[i, 1] += flow_y  # Update y-coordinate based on flow

    # Use SORT tracker for object tracking
    if len(actual_bb) > 0:
        actual_bb[:, 2] += actual_bb[:, 0]  # x2 = x1 + w
        actual_bb[:, 3] += actual_bb[:, 1]  # y2 = y1 + h

        # Run the SORT tracker
        tracked_cars = mot_tracker.update(actual_bb)
    else:
        tracked_cars = mot_tracker.update(np.empty((0, 5)))

    # Draw tracked 2D bounding boxes and optical flow vectors
    for obj in tracked_cars:
        x1, y1, x2, y2, track_id = map(int, obj)

        # Draw bounding box (red)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Store tracking information for output
        track_eval_format.append(f"{frame_number}, {track_id}, {x1}, {y1}, {x2-x1}, {y2-y1}, -1, -1, -1, -1\n")

    # Write the frame with bounding boxes and tracking info to the output video
    out.write(frame)

    # Update previous frame for the next iteration
    prev_frame = frame

# Release resources
cap.release()
out.release()

# Save the tracking results to a text file
write_results_to_txt(track_eval_format, '/ghome/c5mcv01/mcv-c6-2025-team1/week3/src/tracking/s03_with_of.txt')

print("Annotated video with bounding boxes, track IDs, and optical flow vectors saved.")












                
                