from ultralytics import YOLO
import cv2
from utils import * 
import numpy as np

# Load the YOLOv8 model
model = YOLO("yolo12n.pt")

# Path to the input video and annotations
video_path = "/ghome/c3mcv02/mcv-c6-2025-team1/data/AICity_data/train/S03/c010/vdo.avi"
annotations_path = "/ghome/c3mcv02/mcv-c6-2025-team1/data/ai_challenge_s03_c010-full_annotation.xml"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the writer to save the processed video
output_path = "YOLOv12.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Get the list of class names (ensure it matches the model's classes)
class_names = model.names  

# Find the class IDs for "car" and "truck"
car_class_id = [idx for idx, name in class_names.items() if name == "car"][0]
truck_class_id = [idx for idx, name in class_names.items() if name == "truck"][0]

# Read ground truth annotations
gt_boxes = read_annotations(annotations_path)

# Store the precision, recall, and mean average precision metrics for both IoU thresholds
metrics_05 = []  # For IoU threshold 0.5
metrics_075 = []  # For IoU threshold 0.75
detection_file_path = "det_yolov12.txt"
with open(detection_file_path, "w") as f:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if there are no more frames

        # Get the current frame number
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Run inference on the current frame
        results = model(frame)

        # Collect ground truth and predicted boxes
        pred_boxes = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())  # Get the detected class ID

                # Check if the class is either "car" or "truck"
                if class_id == car_class_id or class_id == truck_class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

                    # Add predicted box to list (for mAP calculation) without the confidence
                    pred_boxes.append([x1, y1, x2, y2])
                    
                    conf = box.conf[0].item()  # Confidence score

                    # Format the detection as [frame_id, -1, x1, y1, x2, y2, confidence, -1, -1, -1] (NO HO SE SI CAL)
                    detection_line = f"{frame_number},-1,{x1},{y1},{x2},{y2},{conf},-1,-1,-1\n"

                    # Write the detection to the file
                    f.write(detection_line)

                    # Set label as "Car" and bounding box color to red for both classes
                    label = f"Car"
                    color = (0, 0, 255)  # Red color for bounding box

                    # Draw the bounding box in red
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Add the label in red
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw the GT bounding boxes in green
        if str(frame_number) in gt_boxes:
            gt_frame_boxes = []
            for box in gt_boxes[str(frame_number)]:
                xtl, ytl, xbr, ybr = map(int, [box["xtl"], box["ytl"], box["xbr"], box["ybr"]])
                # Add GT box to list (for mAP calculation)
                gt_frame_boxes.append([xtl, ytl, xbr, ybr])
                # Draw the bounding box in green
                cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
        
        out.write(frame)
        # Calculate mAP for IoU 0.5 and 0.75
        if gt_frame_boxes and pred_boxes:
            mAP_05 = mean_avg_precision(gt_frame_boxes, pred_boxes, iou_threshold=0.5)
            mAP_075 = mean_avg_precision(gt_frame_boxes, pred_boxes, iou_threshold=0.75)

            metrics_05.append(mAP_05)
            metrics_075.append(mAP_075)

        
        # Write the processed frame to the output video
        out.write(frame)

# Release resources
cap.release()
out.release()

# Calculate the average mAP for IoU thresholds 0.5 and 0.75
mAP_05 = np.mean(metrics_05) if metrics_05 else 0
mAP_075 = np.mean(metrics_075) if metrics_075 else 0

print(f"mAP@0.5: {mAP_05:.4f}")
print(f"mAP@0.75: {mAP_075:.4f}")
