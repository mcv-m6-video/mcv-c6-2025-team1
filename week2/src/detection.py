import argparse
from models_files.faster_rcnn import FasterRCNN
from models_files.ssd_vgg16 import SSD_VGG16
from models_files.detr import DETR
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from ultralytics import YOLO
import cv2
import numpy as np
from utils import *


def get_model(model_type: str, model_path: str, box_score_threshold: float) -> any:
    """Get the model based on the model type.

    Args:
        model_type (str): Model type (yolo, faster-rcnn, ssd, detr)
        model_path (str): Path to the model file (ignored for DETR)
        box_score_threshold (float): Box score threshold

    Raises:
        ValueError: If the model type is not supported

    Returns:
        any: Model object
    """
    if model_type == "yolo":
        model = YOLO(model_path)
    elif model_type == "faster-rcnn":
        model = FasterRCNN(box_score_threshold)
    elif model_type == "ssd-vgg16":
        model = SSD_VGG16(box_score_threshold)
    elif model_type == "detr":
        model = DETR(score_threshold=box_score_threshold)  
    else:
        raise ValueError("Model type not supported. Choose 'yolo', 'faster-rcnn', 'ssd-vgg16' or 'detr'.")
    return model


def process_video(model_type, model_path, video_path, output_video_path, annotations_path, box_score_threshold=0.9):
    # Load the YOLOv8 model
    model = get_model(model_type, model_path, box_score_threshold)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the writer to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Get the list of class names (ensure it matches the model's classes)
    # and find the class IDs for "car" and "truck"
    if model_type == "detr":
        class_names = model.categories
    elif model_type == "yolo":
        class_names = model.model.names
    elif model_type == "faster-rcnn":
        class_names = {i: name for i, name in enumerate(model.get_classes())}
    elif model_type == "ssd-vgg16":
        weights = SSD300_VGG16_Weights.DEFAULT
        class_labels = weights.meta["categories"]
        class_names = {i: name for i, name in enumerate(class_labels)}

    try:
        if isinstance(class_names, list):  # If it is a list
            try:
                car_class_id = class_names.index("car")
                truck_class_id = class_names.index("truck")
            except ValueError:
                raise ValueError("Could not find 'car' or 'truck' classes in the model's class names")
        else:  # If it's a dictionary (like in YOLO or Faster-RCNN)
            car_class_id = next(idx for idx, name in class_names.items() if name.lower() == "car")
            truck_class_id = next(idx for idx, name in class_names.items() if name.lower() == "truck")

        print(f"Car class ID: {car_class_id}, Truck class ID: {truck_class_id}")
    except StopIteration:
        raise ValueError("Could not find 'car' or 'truck' classes in the model's class names")

    # Read the ground truth annotations
    gt_boxes = read_annotations(annotations_path)

    # Store the precision, recall, and mean average precision metrics for both IoU thresholds
    metrics_05 = []
    metrics_075 = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get the current frame number
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print("-" * 50)
        print(f"Processing frame {frame_number}")

        # Run inference on the current frame
        if model_type == "yolo":
            results = model(frame)
        elif model_type == "detr":
            results = model.predict(frame)
            print("DETR Results:", results)
        elif model_type == "faster-rcnn":
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = model.predict(frame_gray)
        elif model_type == "ssd-vgg16":
            results = model.predict(frame)

        # Collect predicted boxes
        pred_boxes = []
        if model_type == "yolo":
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())

                    # Check if the class is "car" or "truck"
                    if class_id == car_class_id or class_id == truck_class_id:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Add predicted box to list
                        pred_boxes.append([x1, y1, x2, y2, box.conf[0].item()])
        if model_type == "detr":
            boxes = results['boxes']
            labels = results['labels']
            scores = results['scores']

            for box, label, score in zip(boxes, labels, scores):
                if label.item() == car_class_id or label.item() == truck_class_id:
                    # Scale the box coordinates from [0, 1] to the actual image dimensions
                    x1, y1, x2, y2 = box.tolist()
                    x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
                    pred_boxes.append([x1, y1, x2, y2, score.item()])
                    print("DETR pred_boxes:", pred_boxes)

        else:  # faster-rcnn
            boxes = results[0]['boxes']
            scores = results[0]['scores']
            labels = results[0]['labels']
            
            for box, score, label in zip(boxes, scores, labels):
                if label == car_class_id or label == truck_class_id:
                    x1, y1, x2, y2 = map(int, box.tolist())
                    
                    # Add predicted box to list
                    pred_boxes.append([x1, y1, x2, y2, score.item()])
                
        # Sort the predicted boxes by confidence (highest to lowest)
        pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)

        # Draw the predicted bounding boxes in red
        for x1, y1, x2, y2, conf in pred_boxes:
            label = f"Car"
            color = (0, 0, 255)  # Red
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw the GT bounding boxes in green
        if str(frame_number-1) in gt_boxes:
            gt_frame_boxes = []
            for box in gt_boxes[str(frame_number-1)]:
                xtl, ytl, xbr, ybr = map(int, [box["xtl"], box["ytl"], box["xbr"], box["ybr"]])
                gt_frame_boxes.append([xtl, ytl, xbr, ybr])
                cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
        out.write(frame)

        # Calculate mAP for IoU 0.5 and 0.75
        if gt_frame_boxes and pred_boxes:
            mAP_05 = mean_avg_precision(gt_frame_boxes, [box[:4] for box in pred_boxes], iou_threshold=0.5)
            mAP_075 = mean_avg_precision(gt_frame_boxes, [box[:4] for box in pred_boxes], iou_threshold=0.75)
            print(f"mAP@0.5: {mAP_05:.4f}")
            print(f"mAP@0.75: {mAP_075:.4f}")

            metrics_05.append(mAP_05)
            metrics_075.append(mAP_075)

    # Release resources
    cap.release()
    out.release()

    # Calculate the average mAP for IoU thresholds 0.5 and 0.75
    mAP_05 = np.mean(metrics_05) if metrics_05 else 0
    mAP_075 = np.mean(metrics_075) if metrics_075 else 0

    print(f"mAP@0.5: {mAP_05:.4f}")
    print(f"mAP@0.75: {mAP_075:.4f}")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process a video using one of the provided models (FasterRCNN, YOLO, detr, etc...).")
    
    # Add arguments for the model, input video, and output video
    parser.add_argument("-t", "--model_type", type=str, help="Which model to use (faster-rcnn, yolo, ssd-vgg16, detr)", required=True)
    parser.add_argument("-m","--model_path", type=str, help="Path to the YOLO model file", default=None)
    parser.add_argument("-b", "--box_score_threshold", type=float, help="Box score threshold for FasterRCNN, ssd-vgg16 and detr", default=0.5)
    parser.add_argument("-v","--video_path", type=str, help="Path to the input video file")
    parser.add_argument("-a","--annotation_path", type=str, help="Path to the ground truth annotations")
    parser.add_argument("-o","--output_video_path", type=str, help="Path to save the output video")

    # Parse the arguments
    args = parser.parse_args()
    
    # Get the arguments
    model_type = args.model_type
    model_path = args.model_path
    box_score_threshold = float(args.box_score_threshold)
    video_path = args.video_path
    output_video_path = args.output_video_path
    annotation_path = args.annotation_path
    
    print(f"Processing video using model: {model_type}")
    if model_type == "yolo":
        print(f"Model path: {model_path}")
    print(f"Input video path: {video_path}")
    print(f"Output video path: {output_video_path}")
    print(f"Annotation path: {annotation_path}")
    if model_type in ["faster-rcnn", "ssd-vgg16", "detr"]:
        print(f"Box score threshold: {box_score_threshold}")
        
    # Call the function to process the video
    process_video(model_type, model_path, video_path, output_video_path, annotation_path, box_score_threshold)
