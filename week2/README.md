<h2 align="center">WEEK 2: Tasks</h2>

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
This task focuses on implementing and evaluating various pre-trained object detection models (including YOLOv11, Faster R-CNN,DeTR and SSD) for vehicle detection in traffic surveillance videos. Below are the detailed commands and parameters used for each experiment.

For further detail in the detection script please refer to our help command:

```bash
python3 detection.py --help
```

#### YOLOv11 and YOLOv12
Our implementation leverages the YOLOv11n, YOLOv11x, and YOLOv12n models, pre-trained using the Ultralytics framework, originally trained on the COCO dataset. These models are optimized for general object detection across diverse, real-world scenes. We apply these models directly for inference, specifically targeting car and truck classes for vehicle detection.

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
Our implementation utilizes PyTorch's pre-trained Faster R-CNN model with a ResNet-50 backbone, originally trained on the COCO dataset. The model architecture consists of a Region Proposal Network (RPN) coupled with a detection network, optimized for general object detection across diverse, real-world scenes. We apply these models directly for inference, specifically for vehicle detection in traffic surveillance scenarios.

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

Our implementations utilizes two versions of PyTorch's pre-trained SSD (Single Shot Detector) model —one with a VGG-16 backbone and another with a ResNet-50 backbone. Both models were originally trained on the COCO dataset. In the SSD architecture, the backbone (either VGG-16 or ResNet-50) extracts multi-scale convolutional features from the input image. These feature maps are then fed into additional convolutional layers that directly predict object class probabilities and bounding box offsets in a single forward pass, eliminating the need for a separate region proposal stage. These models are optimized for general object detection across diverse, real-world scenes. We apply them directly for inference, specifically targeting car and truck classes for vehicle detection.

For our experiments, we process each frame at original resolution and set a confidence threshold (default of 0.9) for detection filtering. The model focuses exclusively on the car and truck classes (COCO class ID: 3 and 8).

You can execute the detection pipeline with:

(SSD with VGG-16 backbone)
```bash
python3 detection.py -t ssd-vgg16 \
    -b 0.9 \
    -v /path/to/video/vdo.avi \
    -a /path/to/annotations.xml \
    -o /path/to/output/ssd_vgg16.avi
```

(SSD with ResNet-50 backbone)
```bash
python3 detection.py -t ssd-resnet50 \
    -b 0.9 \
    -v /path/to/video/vdo.avi \
    -a /path/to/annotations.xml \
    -o /path/to/output/ssd_resnet50.avi
```

The detection script processes each video frame independently, generating bounding box predictions with confidence scores. The results are visualized in real-time and saved to an output video file. Performance metrics including mAP@0.50, mAP@0.75, precision and recall are computed and displayed during execution.

#### DETR

Our implementation utilizes the pre-trained DEtection TRansformer (DETR) model with a ResNet-50 backbone, originally trained on the COCO dataset. DETR is an end-to-end object detection model based on transformers, eliminating the need for traditional region proposal networks.  

For our experiments, we process each frame at its original resolution and set a confidence threshold (default of 0.9) to filter detections. The model is optimized for general object detection across diverse, real-world scenes, and we apply it directly for inference, specifically targeting car and truck classes for vehicle detection (COCO class IDs: 3 and 8).  

You can execute the detection pipeline with:  

```bash
python3 detection.py -t detr \
    -b 0.9 \
    -v /path/to/video/vdo.avi \
    -a /path/to/annotations.xml \
    -o /path/to/output/detr.avi
```

The detection script processes each video frame independently, generating bounding box predictions with confidence scores. The results are visualized in real-time and saved to an output video file. Performance metrics including mAP@0.50, mAP@0.75, precision and recall are computed and displayed during execution.


#### Quantitative results
| Model                                  | mAP@0.5 | mAP@0.75 |
|----------------------------------------|--------|---------|
| **YOLOv11x**                           | 0.52   | 0.47    |
| **YOLOv11n**                           | 0.49   | 0.44    |
| **YOLOv12n**                           | 0.47   | 0.44    |
| **Faster R-CNN (conf. 0.5)**           | 0.57   | 0.44    |
| **Faster R-CNN (conf. 0.7)**           | 0.59   | 0.44    |
| **Faster R-CNN (conf. 0.9)**           | 0.45   | 0.39    |
| **SSD (Backbone: ResNet50, conf. 0.1)** | 0.49   | 0.32    |
| **SSD (Backbone: ResNet50, conf. 0.25)**| 0.39   | 0.31    |
| **SSD (Backbone: ResNet50, conf. 0.5)** | 0.32   | 0.30    |
| **SSD (Backbone: VGG16, conf. 0.25)**   | 0.39   | 0.31    |
| **SSD (Backbone: VGG16, conf. 0.5)**    | 0.30   | 0.29    |
| **SSD (Backbone: VGG16, conf. 0.9)**    | 0.16   | 0.16    |
| **DETR (conf. 0.9)**                    | 0.68 | 0.48  |
| **DETR (conf. 0.7)**                    | 0.75 | 0.49  |
| **DETR (conf. 0.5)**                    | **0.78** | **0.49**  |



#### Qualitative results

- **YOLOv11 and YOLOv12:**

- **Faster R-CNN:**

- **DETR:**

To qualitatively evaluate the **DEtection TRansformer (DETR)** model, we conducted an experiment using the `detr.py` script. This script runs a forward pass of the DETR model on the first frame of a given video and visualizes the detections.  

```bash
python3 detr.py
```

The script processes the input frame by:  
- Loading a pre-trained DETR model with a ResNet-50 backbone.  
- Preprocessing the image and running inference to obtain bounding box predictions.  
- Filtering detections to retain only **cars** and **trucks** based on COCO class IDs (3 and 8).  
- Drawing the predicted bounding boxes on the image.  

Below is an example output, where the model successfully detects vehicles in the first frame of the input video:  

![first_frame_detr](https://github.com/user-attachments/assets/46b3662a-695e-4757-8ce8-5a69f6c56401)

- **SSD:**


### Task 1.2: Fine-tuning to our data

### Task 1.3: K-Fold Cross Validation

### Task 2.1: Tracking by overlap

### Task 2.2: Tracking with KF

### Task 2.3: IDF1, HOTA scores

