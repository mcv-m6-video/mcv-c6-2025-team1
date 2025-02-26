import cv2
import numpy as np  
import os
import argparse

from src.utils import *


def get_bounding_box(mask: np.ndarray, output_frame: np.ndarray, aspect_ratio_threshold: float = 1.2, area_threshold: int = 918) -> tuple:
        """Get the bounding box of the mask

        Args:
            mask (np.ndarray): The mask to calculate the bounding box of
            output_frame (np.ndarray): The frame to draw the bounding box on

        Returns:
            tuple: Tuple containing the top-left and bottom-right coordinates of the bounding box, and the output frame
        """
        # Get connected components
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        
        coords = []
        for i in range(1, n_labels):
            x, y, w, h, area = stats[i]
            
            if area < area_threshold:  # Filter out small areas
                continue
            
            # Filter out by aspect ratio
            aspect_ratio = h/w
            if aspect_ratio > aspect_ratio_threshold:
                continue
            
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            coords.append((top_left, bottom_right))

        for (top_left, bottom_right) in coords:
            cv2.rectangle(output_frame, top_left, bottom_right, (0, 0, 255), 2)

        return coords, output_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mask", help="Path to the mask video file from ZBS")
    parser.add_argument("-gt", "--ground_truth", help="Path to the ground truth file")
    parser.add_argument("-v", "--verbose", help="Print extra information", action="store_true", default=False)
    args = parser.parse_args()
    
    mask_path = args.mask
    ground_truth_path = args.ground_truth
    verbose = args.verbose
    
    print("Evaluating ZBS algorithm over the Ground Truth")
    print("-" * 50)
    # Print the paths
    if verbose:
        print("Mask path: ", mask_path)
        print("Ground truth path: ", ground_truth_path)
        
    # Read the ground truth file
    gt_boxes = read_annotations(ground_truth_path)
    
    # Read the mask video
    mask_cap = cv2.VideoCapture(mask_path)
    
    # Write the output video
    output_path = os.path.join(os.path.dirname(mask_path), "output_gt.avi")
    out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), int(mask_cap.get(cv2.CAP_PROP_FPS)), (int(mask_cap.get(3)), int(mask_cap.get(4))))
    
    num_frames = int(mask_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    metrics = []
    while True:
        ret, frame = mask_cap.read()
        if not ret:
            break
        
        # Get the frame number
        frame_number = int(mask_cap.get(cv2.CAP_PROP_POS_FRAMES))
        if verbose:
            print(f"Processing frame {frame_number}/{num_frames}")
        
        # Convert to grayscale and get the binarized frame
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_frame = cv2.threshold(grayscale_frame, 127, 255, cv2.THRESH_BINARY)
        
        # Get the predicted boxes
        pred_boxes, out_frame = get_bounding_box(binary_frame, frame)
        
        # Convert to the same format
        gt_box = gt_boxes.get(str(frame_number + 215), [])
        gt_box = [list(map(int, [box["xtl"], box["ytl"], box["xbr"], box["ybr"]])) for box in gt_box]
        pred_box = [[int(box[0][0]), int(box[0][1]), int(box[1][0]), int(box[1][1])] for box in pred_boxes]
        
        # Compare the predicted boxes with the ground truth
        avg_precision = mean_avg_precision(gt_box, pred_box)
        
        # Print the average precision
        if verbose:
            print("Average Precision: ", avg_precision)
        metrics.append(avg_precision)
        
        # Print GT boxes if they exist
        if gt_boxes:
            try:
                for box in gt_boxes[str(frame_number + 215)]:
                    xtl, ytl, xbr, ybr = int(box["xtl"]), int(box["ytl"]), int(box["xbr"]), int(box["ybr"])
                    cv2.rectangle(out_frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
            except KeyError:
                pass
        
        # Write the frame to the output video
        out_writer.write(out_frame)
        
        print(f"Frame {frame_number}/{num_frames} - Average Precision: {avg_precision}")
        print("-" * 50)
    
    print("Mean Average Precision: ", np.mean(metrics))
    
    # Print mAP of 1606 last frames
    print("Mean Average Precision of the last 1606 (75%) frames: ", np.mean(metrics[-1606:]))
    
    mask_cap.release()
    out_writer.release()