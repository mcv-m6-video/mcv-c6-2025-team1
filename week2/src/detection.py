import argparse
import cv2
import random
import numpy as np

from models_files.faster_rcnn import FasterRCNN
from models_files.ssd_vgg16 import SSD_VGG16
from models_files.ssd_resnet50 import SSD_ResNet50
from models_files.detr import DETR
from torchvision.models.detection import SSD300_VGG16_Weights
from ultralytics import YOLO
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
	elif model_type == "ssd-resnet50":
		model = SSD_ResNet50(box_score_threshold)
	elif model_type == "detr":
		model = DETR(score_threshold=box_score_threshold)
	else:
		raise ValueError("Model type not supported. Choose 'yolo', 'faster-rcnn', 'ssd-vgg16' or 'detr'.")
	return model


def process_video(model_type, model_path, video_path, output_video_path, annotations_path, box_score_threshold=0.9, validation_fold: list[int]=None):
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

	# Frames in the validation fold
	if validation_fold:
		print(f"Validation fold: {validation_fold}")

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
	elif model_type == "ssd-resnet50":
		class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

	try:
		if isinstance(class_names, list):  # If it is a list
			try:
				car_class_id = class_names.index("car")
				truck_class_id = class_names.index("truck")
			except ValueError:
				raise ValueError("Could not find 'car' or 'truck' classes in the model's class names")
		else:  # If it's a dictionary (like in YOLO or Faster-RCNN)
			car_class_id = next(idx for idx, name in class_names.items() if name.lower() == "car")
			try:
				truck_class_id = next(idx for idx, name in class_names.items() if name.lower() == "truck")
			except StopIteration:
				# If truck class is not found, use car class ID
				truck_class_id = car_class_id
				print("Warning: Truck class not found, using car class ID instead")

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

		# Skip frame if it is part of the validation set
		if frame_number - 1 not in validation_fold:
			print(f"Skipping frame {frame_number} as it is part of the validation set")
			continue

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
		elif model_type == "ssd-resnet50":
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
		elif model_type == "detr":
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
		elif model_type == "ssd-resnet50":
			#print(f"results: {results}")
			boxes, labels, scores = results[0]
			print(f"labels: {labels}")
			for box, score, label in zip(boxes, scores, labels):
				if label == (car_class_id+1) or label == (truck_class_id+1):
					x1, y1, x2, y2 = box.tolist()
					x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
					pred_boxes.append([x1, y1, x2, y2, score.item()])
					print("SSD-ResNet50 pred_boxes:", pred_boxes)
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

	return mAP_05, mAP_075


def get_k_folds(video_path: str, strategy: str) -> list[list[int]]:
	"""
	Extract frames from a video according to different strategies.

	Args:
		video_path (str): Path to the video file
		strategy (str): Strategy to use ('A', 'B', or 'C')

	Returns:
		List or List of Lists: Frames according to the selected strategy
	"""
	# Open video and get total frame count
	cap = cv2.VideoCapture(video_path)
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	cap.release()

	if strategy == 'A':
		# Strategy A: Last 75% of the video frames
		start_frame = int(total_frames * 0.25)
		return [list(range(start_frame, total_frames))]
	elif strategy == 'B':
		# Strategy B: 4-fold cross-validation with consecutive frames
		fold_size = total_frames // 4
		validation_folds = []

		for i in range(4):
			# All frame indices
			all_frames = list(range(total_frames))

			# Define the test portion for this fold (25%)
			test_start = i * fold_size
			test_end = (i + 1) * fold_size if i < 3 else total_frames
			test_frames = list(range(test_start, test_end))

			# Validation frames are all frames EXCEPT the test frames (75%)
			validation_frames = [frame for frame in all_frames if frame not in test_frames]
			validation_folds.append(validation_frames)

		return validation_folds
	elif strategy == 'C':
		# Strategy C: 4-fold cross-validation with random frames
		# Get all frame indices and shuffle them
		all_frames = list(range(total_frames))
		random.shuffle(all_frames)

		# Divide into 4 approximately equal groups
		fold_size = total_frames // 4
		groups = []

		for i in range(4):
			start_idx = i * fold_size
			end_idx = (i + 1) * fold_size if i < 3 else total_frames
			groups.append(all_frames[start_idx:end_idx])

		# Create 4 folds, each using a different group as test set
		validation_folds = []
		for i in range(4):
			# All frames except the current group are used for validation
			validation_frames = []
			for j in range(4):
				if j != i:  # Skip the test group
					validation_frames.extend(groups[j])
			validation_folds.append(validation_frames)

		return validation_folds
	else:
		raise ValueError("Strategy must be 'A', 'B', or 'C'")


if __name__ == "__main__":
	# Set up argument parsing
	parser = argparse.ArgumentParser(description="Process a video using one of the provided models (FasterRCNN, YOLO, detr, etc...).")

	# Add arguments for the model, input video, and output video
	parser.add_argument("-t", "--model_type", type=str, help="Which model to use (faster-rcnn, yolo, ssd-vgg16, ssd-resnet50, detr)", required=True)
	parser.add_argument("-m","--model_path", type=str, help="Path to the YOLO model file", default=None)
	parser.add_argument("-b", "--box_score_threshold", type=float, help="Box score threshold for FasterRCNN, ssd-vgg16, ssd-resnet50 and detr", default=0.5)
	parser.add_argument("-v","--video_path", type=str, help="Path to the input video file")
	parser.add_argument("-a","--annotation_path", type=str, help="Path to the ground truth annotations")
	parser.add_argument("-o","--output_video_path", type=str, help="Path to save the output video")
	parser.add_argument("-s", "--strategy", type=str, help="Cross-validation strategy (A, B, or C)", default=None)
	args = parser.parse_args()

	# Get the arguments
	model_type = args.model_type
	model_path = args.model_path
	box_score_threshold = float(args.box_score_threshold)
	video_path = args.video_path
	output_video_path = args.output_video_path
	annotation_path = args.annotation_path
	strategy = args.strategy

	print(f"Processing video using model: {model_type}")
	if model_type == "yolo":
		print(f"Model path: {model_path}")
	print(f"Input video path: {video_path}")
	print(f"Output video path: {output_video_path}")
	print(f"Annotation path: {annotation_path}")
	if model_type in ["faster-rcnn", "ssd-vgg16", "ssd-resnet50", "detr"]:
		print(f"Box score threshold: {box_score_threshold}")

	# Call the function to process the video
	if not strategy:
		process_video(model_type, model_path, video_path, output_video_path, annotation_path, box_score_threshold)
	elif strategy:
		val_folds = get_k_folds(video_path, strategy)

		# Lists to store mAP values for each fold
		map50_scores = []
		map75_scores = []

		# Process each fold and collect results
		for fold_idx, val_fold in enumerate(val_folds):
			print(f"Processing fold {fold_idx + 1}/{len(val_folds)}...")
			map50, map75 = process_video(model_type, model_path, video_path,
			                             output_video_path + f"_fold{fold_idx}.avi", annotation_path,
			                             box_score_threshold, validation_fold=val_fold)

			map50_scores.append(map50)
			map75_scores.append(map75)

			# Optionally print individual fold results
			print(f"Fold {fold_idx + 1} results: mAP@0.50 = {map50:.4f}, mAP@0.75 = {map75:.4f}")

		# Mean values
		mean_map50 = np.mean(map50_scores)
		mean_map75 = np.mean(map75_scores)

		# Standard deviations
		std_map50 = np.std(map50_scores)
		std_map75 = np.std(map75_scores)

		# Print summary statistics
		print("\nCross-validation results:")
		print(f"mAP@0.50: mean = {mean_map50:.4f}, std = {std_map50:.4f}")
		print(f"mAP@0.75: mean = {mean_map75:.4f}, std = {std_map75:.4f}")
