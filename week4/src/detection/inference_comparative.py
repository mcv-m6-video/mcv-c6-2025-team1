import argparse
import time
import os
from ultralytics import YOLO


# Argument parser setup
parser = argparse.ArgumentParser(description="Run YOLO inference on a single image.")
parser.add_argument("--model", type=str, required=True, help="Path to the YOLO model.")
parser.add_argument("--out", type=str, required=True, help="Output directory for results.")
args = parser.parse_args()

# Define the input image path
IMAGE_PATH = '/ghome/c5mcv01/mcv-c6-2025-team1/week4/src/detection/data/images/train/c001/000001.jpg' # Modify this to the desired image file

# Load the YOLO model
model = YOLO(args.model)

# Ensure the output directory exists
os.makedirs(args.out, exist_ok=True)

# Start timing inference
start_time = time.time()

# Perform inference and save results
results_inference = model.predict(source=IMAGE_PATH, save=True, classes=[2],project=args.out)

# Compute inference time
inference_time = time.time() - start_time
print(f"Inference time: {inference_time:.2f} seconds.")


