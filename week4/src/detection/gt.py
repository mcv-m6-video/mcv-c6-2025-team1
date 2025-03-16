import cv2
import os
import argparse

def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def convert_to_yolo(gt_path, output_label_dir, img_width, img_height):
    with open(gt_path, 'r') as f:
        lines = f.readlines()
    
    detections = {}
    for line in lines:
        data = line.strip().split(',')
        frame_id = int(data[0])
        x, y, w, h = map(int, data[2:6])
        
        # Normalitzar coordenades YOLO
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        
        if frame_id not in detections:
            detections[frame_id] = []
        detections[frame_id].append(f"2 {x_center} {y_center} {w_norm} {h_norm}\n")
    
    # Escriure les anotacions
    for frame_id, objects in detections.items():
        label_path = os.path.join(output_label_dir, f"{frame_id:06d}.txt")
        with open(label_path, 'w') as label_file:
            label_file.writelines(objects)

def extract_frames(video_path, output_img_dir, fps):
    os.makedirs(output_img_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, original_fps // fps)  # Saltar frames segons el FPS desitjat
    
    frame_id = 1
    saved_frame_id = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_id % frame_interval == 0:
            img_path = os.path.join(output_img_dir, f"{saved_frame_id:06d}.jpg")
            cv2.imwrite(img_path, frame)
            saved_frame_id += 1
        
        frame_id += 1
    cap.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Path to the video file")
    parser.add_argument("--gt", help="Path to the ground truth file")
    parser.add_argument("--output_images", help="Output directory for images")
    parser.add_argument("--output_labels", help="Output directory for YOLO labels")
    parser.add_argument("--fps", type=int, default=10, help="FPS for frame extraction")
    args = parser.parse_args()
    
    os.makedirs(args.output_labels, exist_ok=True)
    
    img_width, img_height = get_video_resolution(args.video)
    extract_frames(args.video, args.output_images, args.fps)
    convert_to_yolo(args.gt, args.output_labels, img_width, img_height)
    
if __name__ == "__main__":
    main()
