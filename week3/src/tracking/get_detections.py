from ultralytics import YOLO
import cv2
import os

# Paths
base_path = "/ghome/c5mcv01/mcv-c6-2025-team1/data/train/S01"
output_dir = "/ghome/c5mcv01/mcv-c6-2025-team1/week3/detections_yolo_2_1"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load YOLO model
model = YOLO('yolov8x.pt')  # More accurate model

# Classes to detect
class_names = model.names
keep_classes = [name for i, name in class_names.items() if name in ["car", "truck"]]

# Iterate through each folder in S01
for folder in sorted(os.listdir(base_path)):
    folder_path = os.path.join(base_path, folder)
    video_path = os.path.join(folder_path, "vdo.avi")

    if not os.path.exists(video_path):
        print(f"⚠️ Video not found in {folder_path}")
        continue

    # Output file named after the folder
    output_txt = os.path.join(output_dir, f"detections_{folder}.txt")

    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_id = 1

    with open(output_txt, "w") as f:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"✅ Processing completed for {folder}")
                break

            print(f"📍 Processing {folder} - Frame {frame_id}")

            # Run inference
            results = model(frame)

            for result in results:
                for box in result.boxes:
                    cls = int(box.cls)  # Class index
                    class_name = class_names.get(cls, "unknown")

                    if class_name in keep_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                        conf = float(box.conf[0])  # Confidence score
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1

                        # Write to file
                        f.write(f"{frame_id},-1,{x1},{y1},{bbox_width},{bbox_height},{conf:.4f}\n")
                        f.flush()  # Ensure immediate write

            frame_id += 1

    cap.release()

print("🎯 Detection completed for all videos.")
