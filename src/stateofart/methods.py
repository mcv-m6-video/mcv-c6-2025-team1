import cv2
import numpy as np
from src.utils import *


class BackgroundSubtractionProcessor:
    def __init__(self, method='MOG2'):
        """Initialize the background subtractor based on the selected method."""

        if method == 'MOG2':
            self.fgbg = cv2.createBackgroundSubtractorMOG2(history=175, varThreshold=16)
        elif method == 'KNN':
            self.fgbg = cv2.createBackgroundSubtractorKNN(history=200, dist2Threshold=1100.0, detectShadows=True)
        elif method == 'CNT':
            self.fgbg = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=1,maxPixelStability=6)
        elif method == 'GMG':
            self.fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=10,decisionThreshold = 0.98)
        elif method == 'GSOC':
            self.fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC(nSamples = 200,replaceRate = 0.01,propagationRate = 0.7)
        
        else:
            raise ValueError(f"Unsupported background subtraction method: {method}")

    def get_mask(self, frame: np.ndarray, opening_size=3, closing_size=13):
        """Apply morphological operations to the mask."""
        kernel_open = np.ones((opening_size, opening_size), np.uint8)
        kernel_close = np.ones((closing_size, closing_size), np.uint8)
        
        mask = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        return mask

    def get_bounding_box(self, mask: np.ndarray, output_frame: np.ndarray, area_threshold: float=959, aspect_ratio_threshold: float=1.2):
        """Get bounding boxes of connected components in the mask."""
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        
        coords = []
        for i in range(1, n_labels):
            x, y, w, h, area = stats[i]
            
            if area < area_threshold:
                continue
            
            aspect_ratio = h / w
            if aspect_ratio > aspect_ratio_threshold:
                continue
            
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            coords.append((top_left, bottom_right))

        # Draw bounding boxes on the output frame
        for (top_left, bottom_right) in coords:
            cv2.rectangle(output_frame, top_left, bottom_right, (0, 0, 255), 2)

        return coords, output_frame


def process_video(video_path, output_path, mask_path, annotations_path=None, method='MOG2', opening_size=3, closing_size=13):
    cap = cv2.VideoCapture(video_path)
    processor = BackgroundSubtractionProcessor(method=method)  # Create instance with chosen method

    # Create video writer with same specs as the original video
    cap_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(cv2.CAP_PROP_FPS)), 
                              (int(cap.get(3)), int(cap.get(4))))
    mask_out = cv2.VideoWriter(mask_path, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(cv2.CAP_PROP_FPS)), 
                               (int(cap.get(3)), int(cap.get(4))))

    # Read ground truth annotations if available
    if annotations_path:
        gt_boxes = read_annotations(annotations_path)

    metrics = []
    
    # Open the video file
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the starting frame (25% of the total frames)
    start_frame = int(total_frames * 0.25)

    # Seek to the starting frame (25% of the video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(f"Processing frame {frame_number} of total frames {total_frames}...")  # Display the current frame
        
        # Apply selected background subtraction method
        mask = processor.fgbg.apply(frame)
        
        # Apply morphological operation to the mask
        mask = processor.get_mask(mask, opening_size, closing_size)
        
        mask[mask == 127] = 0  # Remove shadows (optional)
        
        # Get bounding boxes
        bounding_box, output_frame = processor.get_bounding_box(mask, frame.copy())  # Using the instance's get_bounding_box method

        # Convert bounding boxes to match GT format
        gt_box = gt_boxes.get(str(frame_number), [])
        gt_box = [list(map(int, [box["xtl"], box["ytl"], box["xbr"], box["ybr"]])) for box in gt_box]
        pred_box = [[int(box[0][0]), int(box[0][1]), int(box[1][0]), int(box[1][1])] for box in bounding_box]

        # Calculate the mean average precision, precision, and recall
        metrics.append(mean_avg_precision(gt_box, pred_box))
        if gt_boxes:
                try:
                    for box in gt_boxes[str(frame_number)]:
                        xtl, ytl, xbr, ybr = int(box["xtl"]), int(box["ytl"]), int(box["xbr"]), int(box["ybr"])
                        cv2.rectangle(output_frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
                except KeyError:
                    pass
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cap_out.write(output_frame)
        mask_out.write(mask)

    print(f"Mean Average Precision: {np.mean(metrics)}")
    
    cap.release()
    cap_out.release()
    mask_out.release()

video_path = '/ghome/c3mcv02/mcv-c6-2025-team1/data/AICity_data/train/S03/c010/vdo.avi'  # Path to input video file
annotations_path = '/ghome/c3mcv02/mcv-c6-2025-team1/data/ai_challenge_s03_c010-full_annotation.xml'  # Path to the annotations file

method = input("Choose background subtraction method (MOG2, KNN, CNT, GMG, GSOC): ").strip()
output_path = f'{method}_video.avi'  # Use f-string to embed method into the output filename
mask_path = f'{method}_mask.avi'  # Similarly, set the mask path
opening_size = int(input("Enter kernel size for opening (default 3): ") or 3)
closing_size = int(input("Enter kernel size for closing (default 13): ") or 13)

# Call the process_video function
process_video(video_path, output_path, mask_path, annotations_path, method=method, opening_size=opening_size, closing_size=closing_size)