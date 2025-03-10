import cv2
import numpy as np
import argparse
import time
from pathlib import Path

import torch
from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import WEIGHTS, TRACKER_CONFIGS

from src.tracking.utils import write_results_to_txt, read_detections_from_txt


def run(args):
    # Read offline detections
    _, detections_vect = read_detections_from_txt(args.detection_file_path)
    
    # Initialize video capture
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize BoxMOT tracker
    assert args.tracking_method in TRACKERS, \
        f"'{args.tracking_method}' is not supported. Supported ones are {TRACKERS}"
    
    tracking_config = TRACKER_CONFIGS / (args.tracking_method + '.yaml')
    tracker = create_tracker(
        args.tracking_method,
        tracking_config,
        args.reid_model, # TODO: Check this...
        args.device,
        args.half,
        args.per_class
    )
    
    # Warm up the tracker model if it has one
    if hasattr(tracker, 'model'):
        tracker.model.warmup()
    
    # Process each frame of the video
    track_eval_format = []
    
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
        
        if len(frame_detections) > 0:
            # TODO: Convert to the format expected by BoxMOT
            # BoxMOT expects [x1, y1, x2, y2, conf, class_id]
            dets_for_tracker = np.array(frame_detections)
            dets_for_tracker_xyxy = np.copy(dets_for_tracker)
            # Convert from [x, y, w, h] to [x1, y1, x2, y2]
            dets_for_tracker_xyxy[:, 2] = dets_for_tracker_xyxy[:, 0] + dets_for_tracker_xyxy[:, 2]
            dets_for_tracker_xyxy[:, 3] = dets_for_tracker_xyxy[:, 1] + dets_for_tracker_xyxy[:, 3]
            
            # TODO: Add confidence scores (using 0.9 as default) and class_id (0 for vehicles)
            conf = np.ones((dets_for_tracker_xyxy.shape[0], 1)) * 0.9
            class_id = np.zeros((dets_for_tracker_xyxy.shape[0], 1))
            dets_for_tracker_xyxy = np.hstack((dets_for_tracker_xyxy, conf, class_id))
            
            # Run the tracker
            init_time_track = time.time()
            tracker_outputs = tracker.update(dets_for_tracker_xyxy, frame)  # frame is needed for some trackers
            print(f"Tracking processed in {time.time() - init_time_track} seconds.")
            
            # tracker_outputs contains [x1, y1, x2, y2, track_id, class_id, conf]
            for output in tracker_outputs:
                x1, y1, x2, y2, track_id = int(output[0]), int(output[1]), int(output[2]), int(output[3]), int(output[4])
                w, h = x2 - x1, y2 - y1
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Store tracking information for output (in MOTChallenge format)
                track_eval_format.append(f"{frame_number},{track_id},{x1},{y1},{w},{h},1,-1,-1,-1\n")
                
        else:
            # If no detections, still call update with empty detection array
            tracker.update(np.empty((0, 6)), frame)
            
        # Write frame to output video
        out.write(frame)
        print(f"Frame processed in {time.time() - init_time_frame} seconds.")
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
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
    parser.add_argument("--reid_model", type=Path, default='osnet_x0_25_msmt17.pt', help="Path to reid model")
    parser.add_argument("--device", default="cuda", help="Whether to use cuda device or cpu", choices=["cuda", "cpu"])
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--per_class", action="store_true", help="not mix up classes when tracking")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        run(args)