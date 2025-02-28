<h2 align="center">WEEK 1: Tasks</h2>

## Table of Contents

- [Project Structure W2](#project-structure-w2)
- [Task 1.1: Off-the-shelf](#task-11-off-the-shelf)
- [Task 1.2: Fine-tuning to our data](#task-12-fine-tuning-to-our-data)
- [Task 1.3: K-Fold Cross Validation](#task-13-k-fold-cross-validation)
- [Task 2.1: Tracking by overlap](#task-21-tracking-by-overlap)
- [Task 2.2: Tracking with KF](#task-22-tracking-with-kf)
- [Task 2.3: IDF1, HOTA scores](#task-23-idf1-hota-scores)


### Project Structure W2

    week2/
    ├── checkpoints/
    ├── output_videos/
    ├── src/
    │   ├── models/
    │   │   ├── faster_rcnn.py
    │   │   ├── ssd.py
    │   │   └── ...
    │   ├── detection.py
    │   ├── utils.py
    │   └── ...
    
### Task 1.1: Off-the-shelf
This task focuses on implementing and evaluating various pre-trained object detection models (including YOLOv11, Faster R-CNN, and SSD) for vehicle detection in traffic surveillance videos. Below are the detailed commands and parameters used for each experiment.

For further detail in the detection script please refer to our help command:

```bash
python3 detection.py --help
```

#### YOLOv11 and YOLOv12
Our implementation leverages the YOLOv11n, YOLOv11x, and YOLOv12n models, pre-trained using the Ultralytics framework, originally trained on the COCO dataset. These models are optimized for vehicle detection, focussing exclusively on the car and the truck classes.

The YOLOv11n and YOLOv12n models are lightweight versions designed for real-time applications with high computational efficiency, while YOLOv11x is larger and more precise model at the cost of higher computational demand.

You can execute the detection pipeline with:

```bash
python3 detection.py -t yolo \
    -m /path/to/model/yolo11n.pt
    -v /path/to/video/vdo.avi \
    -a /path/to/annotations.xml \
    -o /path/to/output/YOLOv11n.avi
```
The detection script processes each video frame independently, generating bounding box predictions with confidence scores. The results are displayed in real-time and saved to an output video file. Performance metrics, including mAP@0.50, mAP@0.75, are computed and displayer during execution. 


#### Faster R-CNN
Our implementation utilizes PyTorch's pre-trained Faster R-CNN model with a ResNet-50 backbone, originally trained on the COCO dataset. The model architecture consists of a Region Proposal Network (RPN) coupled with a detection network, specifically optimized for vehicle detection in traffic surveillance scenarios.

For our experiments, we process each frame at original resolution and set a confidence threshold (default of 0.9) for detection filtering. The model focuses exclusively on the car and truck classes (COCO class ID: 3 and 8).

You can execute the detection pipeline with:

```bash
python3 detection.py -t faster-rcnn \
    -b 0.9 \
    -v /path/to/video/vdo.avi \
    -a /path/to/annotations.xml \
    -o /path/to/output/faster_rcnn.avi
```

The detection script processes each video frame independently, generating bounding box predictions with confidence scores. The results are visualized in real-time and saved to an output video file. Performance metrics including mAP@0.50, mAP@0.75, precision and recall are computed and displayed during execution.

#### SSD
To be done.

#### Qualitative results

#### Quantitative results
| Model                | mAP@0.5 | mAP@0.75 |
|----------------------|--------|---------|
| **YOLOv11x**        | 0.52 | 0.47  |
| **YOLOv11n**        | 0.49 | 0.44  |
| **YOLOv12n**        | 0.47 | 0.44  |
| **Faster R-CNN (conf. 0.5)** | 0.57 | 0.44  |
| **Faster R-CNN (conf. 0.7)** | 0.59 | 0.44  |
| **Faster R-CNN (conf. 0.9)** | 0.45 | 0.39  |


### Task 1.2: Fine-tuning to our data

### Task 1.3: K-Fold Cross Validation

### Task 2.1: Tracking by overlap

### Task 2.2: Tracking with KF

### Task 2.3: IDF1, HOTA scores

