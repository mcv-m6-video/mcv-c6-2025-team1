from ultralytics import YOLO
import cv2
import os

# Paths
video_path = "/ghome/c3mcv02/mcv-c6-2025-team1/data/AICity_data/train/S03/c010/vdo.avi"
model_path = "/ghome/c3mcv02/mcv-c6-2025-team1/week2/yolo_finetuning_results/backbone_frozen/yolo_train_fold_random_0/train/weights/best.pt"
output_txt = "detections_yolo_backbone.txt"

# Load model
model = YOLO(model_path)

# Open video
cap = cv2.VideoCapture(video_path)
frame_id = 1  # Start counting from 1

# Get class names from the YOLO model
class_names = model.names
keep_classes = [name for i, name in class_names.items() if name in ["car", "truck"]]

with open(output_txt, "w") as f:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        
        print(f"Processing frame {frame_id}")  # Print current frame
        
        # Run inference
        results = model(frame)
        print(f"Number of detections in frame {frame_id}: {len(results[0].boxes)}")
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls)  # Get class index
                class_name = class_names.get(cls, "unknown")
                if class_name in keep_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                    conf = float(box.conf[0])  # Confidence score
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    print(f"Detected: frame {frame_id}, class {class_name}, bbox ({x1}, {y1}, {bbox_width}, {bbox_height}), conf {conf:.4f}")
                    
                    # Write to file
                    f.write(f"{frame_id},-1,{x1},{y1},{bbox_width},{bbox_height},{conf:.4f}\n")
                    f.flush()  # Force write to file
        
        frame_id += 1

cap.release()
print(f"Inference completed. Results saved to {output_txt}")