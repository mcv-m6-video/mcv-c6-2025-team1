import os
import cv2
import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm
from ultralytics import YOLO


def run_inference_and_create_video(
    model_path, 
    images_dir, 
    output_video_path, 
    conf_threshold=0.3, 
    fps=10, 
    classes=None
):
    """
    Run YOLO inference on images and create a video with bounding boxes.
    
    Args:
        model_path (str): Path to the YOLO model (.pt file)
        images_dir (str): Directory containing images
        output_video_path (str): Output video file path
        conf_threshold (float): Confidence threshold for detections
        fps (int): Frames per second for output video
        classes (list, optional): Filter for specific classes. Defaults to None.
    """
    # Load the YOLO model
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    # Get image files
    image_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_formats:
        image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
    
    image_files = sorted(image_files)  # Sort to ensure frames are in order
    
    if not image_files:
        raise FileNotFoundError(f"No image files found in {images_dir}")
    
    print(f"Found {len(image_files)} images to process")
    
    # Get dimensions from first image
    first_img = cv2.imread(image_files[0])
    height, width = first_img.shape[:2]
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'XVID'
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing frames"):
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        
        # Run YOLO inference
        results = model(img, conf=conf_threshold, classes=classes)
        
        # Visualize results on the frame
        annotated_frame = results[0].plot()
        
        # Write the frame to video
        video_writer.write(annotated_frame)
    
    # Release the video writer
    video_writer.release()
    print(f"Video saved to {output_video_path}")


def main():
    # Define paths
    BASE_DIR = '/ghome/c3mcv02/mcv-c6-2025-team1'
    MODEL_PATH = '/ghome/c3mcv02/mcv-c6-2025-team1/week2/src/runs/detect/train5/weights/best.pt'
    IMAGES_DIR = f'{BASE_DIR}/week2/dataset/val/images'  # or any path with your images
    OUTPUT_VIDEO = f'{BASE_DIR}/week2/output_videos/finetuned_results_strategyA.avi'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    
    # Run inference and create video
    run_inference_and_create_video(
        model_path=MODEL_PATH,
        images_dir=IMAGES_DIR,
        output_video_path=OUTPUT_VIDEO,
        conf_threshold=0.25,
        fps=10,
        classes=None  # Set to [0] if you only want to detect the first class (usually 'car' in your case)
    )
    print(f"Processing completed. Video saved to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()