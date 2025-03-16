from ultralytics import YOLO
model = YOLO('/ghome/c5mcv01/mcv-c6-2025-team1/week4/src/detection/yolo11x.pt')
results = model.train(data='/ghome/c5mcv01/mcv-c6-2025-team1/week4/src/detection/data/data.yaml', epochs=50, patience=5,batch=0.85,imgsz=1024,project='finetune',fliplr=0.0)