from ultralytics import YOLO


# Load the model and train
model = YOLO('/ghome/c5mcv01/mcv-c6-2025-team1/week4/src/detection/yolo11x.pt')
results = model.train(data='/ghome/c5mcv01/mcv-c6-2025-team1/week4/src/detection/data/data.yaml', epochs=50, patience=5,batch=0.85,imgsz=1024,freeze=10,project='finetune_backbone',fliplr=0.0)