import cv2
import numpy as np
import argparse
import time

from src.tracking.utils import write_results_to_txt, read_detections_from_txt
from src.tracking.sort import Sort
from src.optical_flow.of import OpticalFlow


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalman-filter algorithm for tracking with optical flow.")
    parser.add_argument("-d", "--detection_file_path", help="Path to the object detections TXT file.", required=True, type=str)
    parser.add_argument("-v", "--video_path", help="Path to the video file.", type=str, required=True)
    parser.add_argument("-ov", "--output_video_path", help="Path to the output video.", required=True, type=str)
    parser.add_argument("-o", "--output_path", help="Path to TXT file where the results will be stored", required=True, type=str)
    parser.add_argument("-m", "--of_model", help="Optical flow model to use.", required=False, default="rpknet", type=str, choices=["pyflow", "diclflow", "memflow", "rapidflow", "rpknet", "dip"])
    parser.add_argument("--max_age", required=False, default=21, type=int, help="Max age for SORT.")
    parser.add_argument("--min_hit", help="Min hit for SORT.", required=False, default=2, type=int)
    parser.add_argument("--iou_threshold", help="IoU threshold for SORT.", required=False, type=float, default=0.2)
    args = parser.parse_args()
    
    
    #detections, detections_vect = read_detections_from_txt('/ghome/c5mcv01/mcv-c6-2025-team1/week3/src/tracking/detections_yolo.txt')  
    detections, detections_vect = read_detections_from_txt(args.detection_file_path)

    # Create instance of the SORT tracker (default params: max_age=1, min_hits=3, iou_threshold=0.3)
    mot_tracker = Sort(max_age = args.max_age, min_hits=args.min_hit, iou_threshold=args.iou_threshold) 

    # Open the video
    # cap = cv2.VideoCapture("/ghome/c5mcv01/mcv-c6-2025-team1/data/AICity_data/train/S03/c010/vdo.avi")
    cap = cv2.VideoCapture(args.video_path)

    # Get video information (frame size, fps, etc.)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output_video_path, fourcc, fps, (frame_width, frame_height))

    # Initialize variables for optical flow
    prev_frame = None
    prev_gray = None
    prev_points = None
    
    # Define optical flow model
    optical_flow = OpticalFlow(args.of_model)

    # Process each frame of the video
    track_eval_format = []
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for optical flow calculation
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Get the current frame number
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print("-" * 50)
        print(f"Processing frame {frame_number}")
        init_time_frame = time.time()
        
        actual_bb = [detection[1:5] for detection in detections_vect if detection[0] == frame_number]  
        actual_bb = np.array(actual_bb)

        if prev_frame is not None:
            # Use optical flow to track motion between frames
            init_time_of = time.time()
            flow = optical_flow.compute_flow(prev_frame, frame)
            print(f"Optical flow computed in {time.time() - init_time_of} seconds.")
            
            # Use the optical flow to adjust bounding box positions
            for i, bbox in enumerate(actual_bb):
                x1, y1, w, h = bbox
                flow_x = flow[int(y1), int(x1)][0]  # Horizontal flow component at top-left corner
                flow_y = flow[int(y1), int(x1)][1]  # Vertical flow component at top-left corner
                actual_bb[i, 0] += flow_x 
                actual_bb[i, 1] += flow_y

        # Use SORT tracker for object tracking
        init_time_sort = time.time()
        if len(actual_bb) > 0:
            actual_bb[:, 2] += actual_bb[:, 0]  # x2 = x1 + w
            actual_bb[:, 3] += actual_bb[:, 1]  # y2 = y1 + h

            # Run the SORT tracker
            tracked_cars = mot_tracker.update(actual_bb)
        else:
            tracked_cars = mot_tracker.update(np.empty((0, 5)))
        print(f"SORT processed in {time.time() - init_time_sort} seconds.")

        # Draw tracked 2D bounding boxes and optical flow vectors
        for obj in tracked_cars:
            x1, y1, x2, y2, track_id = map(int, obj)

            # Draw bounding box (red)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame_bgr, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Store tracking information for output
            track_eval_format.append(f"{frame_number}, {track_id}, {x1}, {y1}, {x2-x1}, {y2-y1}, -1, -1, -1, -1\n")

        # Write the frame with bounding boxes and tracking info to the output video
        out.write(frame_bgr)

        # Update previous frame for the next iteration
        prev_frame = frame
        print(f"Frame processed in {time.time() - init_time_frame} seconds.")

    # Release resources
    cap.release()
    out.release()

    # Save the tracking results to a text file
    write_results_to_txt(track_eval_format, args.output_path)
    print("Annotated video with bounding boxes, track IDs, and optical flow vectors saved.")
