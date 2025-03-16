import cv2
import numpy as np
import argparse
import time

from src.tracking.utils import write_results_to_txt, read_detections_from_txt



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

def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) of two boxes.
    Boxes are expected in [x1, y1, x2, y2] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def fuse_box(pred_box, det_box, alpha=0.5):
    """
    Fuse two boxes (predicted and detected) using a weighted average.
    Each box is in [x1, y1, x2, y2] format.
    alpha is the weight for the detection.
    """
    fused = alpha * np.array(det_box) + (1 - alpha) * np.array(pred_box)
    return fused

def box_to_corners(box):
    """
    Convert a box from [x, y, w, h] to [x1, y1, x2, y2].
    """
    x, y, w, h = box
    return [x, y, x + w, y + h]

def corners_to_box(corners):
    """
    Convert a box from [x1, y1, x2, y2] to [x, y, w, h].
    """
    x1, y1, x2, y2 = corners
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]

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
    parser.add_argument("--alpha", help="Alpha for predicted-detected 2DBB fusion.", required=False, type=float, default=0.5)
    parser.add_argument("--pred_method", help="Prediction method used to select the OF vector for a 2DBB.", required=False, type=str, default="weighted_avg")
    parser.add_argument("--sigma", help="Sigma factor required for the weighted average of the OF.", required=False, type=float, default=4)
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
    tracked_bb = []
    
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
            pred_bb = tracked_bb
            for i, bbox in enumerate(tracked_bb):
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                # Extract the OF region of interest (ROI) --> region corresponding to the 2DBB
                roi_flow = flow[int(y1):int(y2), int(x1):int(x2)]

                if args.pred_method == "mean":
                    bb_flow = roi_flow.mean(axis=(0, 1))
                elif args.pred_method == "max":
                    bb_flow = roi_flow.max(axis=(0, 1))
                elif args.pred_method == "gauss_weighted_avg":
                    weights = compute_weights(roi_flow.shape[:2], args.sigma)  # Shape (H, W)
                    weights = np.expand_dims(weights, axis=-1)  # Shape (H, W, 1), so it can broadcast over (H, W, 2)
                    bb_flow = np.average(roi_flow, axis=(0, 1), weights=weights)
                elif args.pred_method == "median":
                    bb_flow = np.median(roi_flow, axis=(0, 1))
                    

                if not np.any(np.isnan(bb_flow)):
                    pred_bb[i, 0] += bb_flow[0]
                    pred_bb[i, 1] += bb_flow[1]
        else:
            pred_bb = actual_bb
        
        # Fuse predicted and detected 2DBB
        if len(actual_bb) > 0 and len(pred_bb) > 0:
            fused_bb = match_and_fuse(pred_bb, actual_bb, iou_threshold=args.iou_threshold, alpha=args.alpha)
        elif len(actual_bb) > 0:
            fused_bb = [box_to_corners(box) for box in actual_bb]
        #elif len(pred_bb) > 0:
        #    fused_bb = [box_to_corners(box) for box in pred_bb]
        else:
            fused_bb = np.empty((0, 4))

        # Use SORT tracker for object tracking taking the predicted actual 2DBB from OF info
        init_time_sort = time.time()
        if len(fused_bb) > 0:
            #prev_bb[:, 2] += prev_bb[:, 0]  # x2 = x1 + w
            #prev_bb[:, 3] += prev_bb[:, 1]  # y2 = y1 + h

            # Run the SORT tracker
            tracked_cars = mot_tracker.update(fused_bb)
        else:
            tracked_cars = mot_tracker.update(np.empty((0, 5)))
        print(f"SORT processed in {time.time() - init_time_sort} seconds.")

        # Draw tracked 2D bounding boxes and optical flow vectors
        tracked_bb = []
        for obj in tracked_cars:
            x1, y1, x2, y2, track_id = map(int, obj)

            # Draw bounding box (red)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame_bgr, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Store tracking information for output
            track_eval_format.append(f"{frame_number}, {track_id}, {x1}, {y1}, {x2-x1}, {y2-y1}, -1, -1, -1, -1\n")
            tracked_bb.append([x1, y1, x2-x1, y2-y1])
            #print(f"Tracked cars: {tracked_bb}")

        # Write the frame with bounding boxes and tracking info to the output video
        out.write(frame_bgr)

        # Update previous frame for the next iteration
        prev_frame = frame
        tracked_bb = np.array(tracked_bb)
        print(f"Frame processed in {time.time() - init_time_frame} seconds.")

    # Release resources
    cap.release()
    out.release()

    # Save the tracking results to a text file
    write_results_to_txt(track_eval_format, args.output_path)
    print("Annotated video with bounding boxes, track IDs, and optical flow vectors saved.")
