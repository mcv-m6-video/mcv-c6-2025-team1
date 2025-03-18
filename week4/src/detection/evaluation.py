import argparse
from ultralytics import YOLO


# Argument parser setup
parser = argparse.ArgumentParser(description="Validate a YOLO model.")
parser.add_argument("--m", type=str, required=True, help="Path to the YOLO model file.")
args = parser.parse_args()

# Load the model
model = YOLO(args.m) 
DATASET_PATH = "/ghome/c5mcv01/mcv-c6-2025-team1/week4/src/detection/data/data.yaml"

# Run validation
results = model.val(data=DATASET_PATH)

# Print specific metrics
print("Average precision for all classes:", results.box.all_ap)
print("Average precision:", results.box.ap)
print("Average precision at IoU=0.50:", results.box.ap50)
print("F1 score:", results.box.f1)
print("Mean average precision:", results.box.map)
print("Mean average precision at IoU=0.50:", results.box.map50)
print("Mean average precision at IoU=0.75:", results.box.map75)
print("Mean precision:", results.box.mp)
print("Mean recall:", results.box.mr)
print("Precision:", results.box.p)
print("Recall:", results.box.r)


