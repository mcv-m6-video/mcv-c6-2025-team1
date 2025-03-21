import cv2
import numpy as np
import argparse
import time
from pathlib import Path

import torch
from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import TRACKER_CONFIGS

from src.tracking.utils import write_results_to_txt, read_detections_from_txt, box_to_corners, compute_iou


def fuse_box(pred_box, det_box, alpha=0.5):
    """
    Fuse two boxes (predicted and detected) using a weighted average.
    Each box is in [x1, y1, x2, y2] format.
    alpha is the weight for the detection.
    """
    fused = alpha * np.array(det_box) + (1 - alpha) * np.array(pred_box)
    return fused

def match_and_fuse(pred_boxes, det_boxes, iou_threshold=0.3, alpha=0.5):
    """
    Match predicted boxes with detection boxes and fuse them if IoU exceeds threshold.
    Assumes input boxes are in [x, y, w, h] format.
    Returns fused boxes in [x1, y1, x2, y2] format.
    """
    # Convert boxes to [x1, y1, x2, y2] format.
    pred_corners = [box_to_corners(box) for box in pred_boxes]
    det_corners = [box_to_corners(box) for box in det_boxes]

    fused_boxes = []
    used_det = set()
    # For each predicted box, find the best matching detection.
    for i, pred in enumerate(pred_corners):
        best_iou = 0
        best_det = None
        best_j = -1
        for j, det in enumerate(det_corners):
            iou = compute_iou(pred, det)
            if iou > best_iou:
                best_iou = iou
                best_det = det
                best_j = j
        if best_iou >= iou_threshold and best_j not in used_det:
            fused_box = fuse_box(pred, best_det, alpha)
            fused_boxes.append(fused_box)
            used_det.add(best_j)
        # else:
        # No good detection found; use predicted box.
        #   fused_boxes.append(pred)

    # Optionally, add any detection that was not matched.
    for j, det in enumerate(det_corners):
        if j not in used_det:
            fused_boxes.append(det)
    return np.array(fused_boxes)

def compute_weights(roi_bb, sigma_factor):
    """
    Compute Gaussian weights based on the distance from the center of the ROI (2DBB area).
    Weights decrease as the distance from the center increases, following a Gaussian distribution.
    """
    h, w = roi_bb[:2]
    # Create a grid of (x, y) coordinates
    y_indices, x_indices = np.indices((h, w))
    # Calculate the center coordinates
    x_c, y_c = w / 2, h / 2
    # Compute the squared Euclidean distance from the center
    sq_dist = (x_indices - x_c) ** 2 + (y_indices - y_c) ** 2
    # Compute the std dev. for the Gaussian function
    sigma = np.sqrt(h**2 + w**2) / sigma_factor
    # Compute the Gaussian weights
    weights = np.exp(-sq_dist / (2 * sigma**2))
    return weights

def run(args):
    # Read offline detections
    _, detections_vect = read_detections_from_txt(args.detection_file_path)
    
    # Initialize video capture
    cap = cv2.VideoCapture(args.video_path)
    
    # Get the information (frame size, fps, etc.)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize BoxMOT tracker
    assert args.tracking_method in TRACKERS, \
        f"'{args.tracking_method}' is not supported. Supported ones are {TRACKERS}"
    
    # Get the tracking config file
    tracking_config = TRACKER_CONFIGS / (args.tracking_method + '.yaml')
    if args.config_path:
        tracking_config = args.config_path
        
    tracker = create_tracker(
        args.tracking_method,
        tracking_config,
        args.reid_model, # TODO: Check this...
        0 if args.device == "cuda" else "cpu",
        args.half,
        args.per_class
    )
    
    # Warm up the tracker model if it has one
    if hasattr(tracker, 'model'):
        tracker.model.warmup()
    
    # Process each frame of the video
    track_eval_format = []
    
    prev_frame = None
    
    # Store positions of tracked bounding boxes
    box_positions = {}  # Format: {track_id: [(x, y), ...]}

    # Number of frames to check if the car is stationary
    N = 10  # Adjust as needed (10 frames = 1 second at 10 FPS)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get the current frame number
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print("-" * 50)
        print(f"Processing frame {frame_number}")
        init_time_frame = time.time()
        
        # Get detections for current frame
        frame_detections = [detection[1:5] for detection in detections_vect if detection[0] == frame_number]
        actual_bb = np.array(frame_detections)
            
        # Fuse predicted and detected 2DBB
        if len(actual_bb):
            fused_bb = [box_to_corners(box) for box in actual_bb]
        else:
            fused_bb = np.empty((0, 4))
        
        if len(fused_bb) > 0:
            # TODO: Convert to the format expected by BoxMOT
            # BoxMOT expects [x1, y1, x2, y2, conf, class_id]
            dets_for_tracker = np.array(fused_bb)
            dets_for_tracker_xyxy = np.copy(dets_for_tracker)
            
            # TODO: Add confidence scores (using 0.9 as default) and class_id (0 for vehicles)
            conf = np.ones((dets_for_tracker_xyxy.shape[0], 1)) * 0.9
            class_id = np.zeros((dets_for_tracker_xyxy.shape[0], 1))
            dets_for_tracker_xyxy = np.hstack((dets_for_tracker_xyxy, conf, class_id))
            
            # Run the tracker
            init_time_track = time.time()
            tracker_outputs = tracker.update(dets_for_tracker_xyxy, frame)  # frame is needed for some trackers
            print(f"Tracking processed in {time.time() - init_time_track} seconds.")
            
            # tracker_outputs contains [x1, y1, x2, y2, track_id, class_id, conf]
            tracked_bb = []
            for output in tracker_outputs:
                x1, y1, x2, y2, track_id = int(output[0]), int(output[1]), int(output[2]), int(output[3]), int(output[4])
                w, h = x2 - x1, y2 - y1
                
                # Store the current position of the tracked bounding box
                if track_id not in box_positions:
                    box_positions[track_id] = []
                box_positions[track_id].append((x1, y1))
                
                # Check if the vehicle has moved significantly in the last N frames
                if len(box_positions[track_id]) > N:
                    last_positions = box_positions[track_id][-N:]
                    distances = [np.linalg.norm(np.array(last_positions[i]) - np.array(last_positions[i+1])) for i in range(N-1)]
                    if all(d < 30 for d in distances):  
                        continue  # If no significant movement, ignore this vehicle
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Store tracking information for output (in MOTChallenge format)
                track_eval_format.append(f"{frame_number},{track_id},{x1},{y1},{w},{h},1,-1,-1,-1\n")
                tracked_bb.append([x1, y1, x2-x1, y2-y1])
        else:
            # If no detections, still call update with empty detection array
            tracker.update(np.empty((0, 6)), frame)
            
        # Write frame to output video
        out.write(frame)
        
        print(f"Frame processed in {time.time() - init_time_frame} seconds.")
    
    # Release resources
    cap.release()
    out.release()
    
    # Save the tracking results to a text file
    write_results_to_txt(track_eval_format, args.output_path)
    print(f"Tracking results saved to {args.output_path}")
    print(f"Annotated video saved to {args.output_video_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="BoxMOT tracker with offline detections")
    parser.add_argument("-d", "--detection_file_path", help="Path to the object detections TXT file.", required=True, type=str)
    parser.add_argument("-v", "--video_path", help="Path to the video file.", type=str, required=True)
    parser.add_argument("-ov", "--output_video_path", help="Path to the output video.", required=True, type=str)
    parser.add_argument("-o", "--output_path", help="Path to TXT file where the results will be stored", required=True, type=str)
    parser.add_argument("-m", "--tracking_method", type=str, default="deepocsort", help="Tracking method: deepocsort, botsort, strongsort, ocsort, bytetrack")
    parser.add_argument("-c", "--config_path", help="Path to the configuration for tracking model.", type=str, required=False, default=None)
    parser.add_argument("--reid_model", type=Path, default='osnet_x0_25_msmt17.pt', help="Path to reid model")
    parser.add_argument("--device", default="cuda", help="Whether to use cuda device or cpu", choices=["cuda", "cpu"])
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--per_class", action="store_true", help="not mix up classes when tracking")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        run(args)
