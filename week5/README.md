<h2 align="center">WEEK 5: Task 1 - Ball Action Classification</h2>

> [!IMPORTANT]
> This week of the project has been uploaded in a fork repository in a different place. In order to access the code and model settings, please use this command to pull the data from that repository:
> `git submodule update --init --recursive`

## Table of Contents
- [Introduction](#introduction)
- [Dataset and Pre-processing](#dataset-and-pre-processing)
- [Baseline Model](#baseline-model)
- [Implemented Models](#implemented-models)
- [Training Details](#training-details)
- [Evaluation](#evaluation)
- [How to Run](#how-to-run)
- [Results](#results)
- [References](#references)

## Introduction
The objective of this task is to familiarize ourselves with the dataset and the provided starter code and improve upon the baseline end-to-end action classification model for ball-related activities.

The repository contains two main scripts:
- **extract_frames_snb.py**: Extracts frames from videos (default resolution: 398x224, 25 FPS).
- **main_classification.py**: Trains and evaluates the baseline classification model.

## Dataset and Pre-processing
- Clips consist of **L = 50 frames**.
- Frames are sampled with a **stride of 2**, reducing the frame rate from **25 FPS to 12.5 FPS**.
- Clips are extracted with a **90% temporal overlap** for training/validation and **0% overlap** for testing.

## Baseline Model
- **Backbone**: RegNet-Y (200MF) without the last FC layer to produce D-dimensional frame-level features.
- **Feature Aggregation**: Max-pooling along the temporal dimension, resulting in a D-dimensional clip representation.
- **Classification Head**: Fully Connected (FC) layer + Sigmoid activation before loss computation.

## Implemented Models
We explored various architectures and modifications to improve the baseline performance:
- **Baseline**
- **Baseline + LSTM**
- **Baseline + Graph**
- **Baseline + LSTM + Attention**
- **Baseline + LSTM(2L) + Attention**
- **Baseline + Transformers**
- **3D-R18**
- **3D-R18 + LSTM + Attention**
- **3D-R18 + LSTM + Attention + Focal Loss**
- **mViT**
- **x3D_S**
- **x3D_M (Frozen)**
- **x3D_M (Frozen) + LSTM + Attention**
- **x3D_M**
- **x3D_M + LSTM + Attention**
- **x3D_M + LSTM + Attention + Focal Loss**
- **x3D_M + LSTM + Attention + Stride 3, 4, 5, 6**
- **x3D_M + LSTM + Attention + Clip Length 80, 100**
- **x3D_M + LSTM + Attention + L2 Regularization (1e-4)**
- **x3D_M + LSTM + Attention + Label Smoothing**
- **x3D_M + LSTM + Attention + 1.5× Image Size**

## Training Details
- **Loss Function**: Binary Cross-Entropy (BCE) for multi-label classification.
- **Batch Size**: 4 clips (~5.15 GB of GPU memory usage).
- **Epochs**: 15 (each epoch lasts approximately 6-7 minutes).

## Evaluation
- We evaluate the model using **Average Precision (AP)** from scikit-learn.
- Reported metrics:
  - **Per-class AP**
  - **Mean AP (mAP)**
  - **mAP@12**: Mean AP for all classes.
  - **mAP@10**: Mean AP for the first 10 classes, as recommended by the professor.

## How to Run
To train and evaluate the baseline model, run:
```bash
python3 main_classification.py --model <model_name>
```
Where `<model_name>` should match the name of a configuration file in the `config` directory (e.g., `baseline.json`). Example command for running the baseline model:
```bash
python3 main_classification.py --model baseline
```
Additionally, you need to modify the following line in `main_classification.py` to use the desired model implementation from the `model` directory:
```python
from model.model_classification import Model
```
For example, to use `model_classification_x3d_lstm_attention.py`, modify it to:
```python
from model.model_classification_x3d_lstm_attention import Model
```
For additional configuration options, refer to the README inside the `config` directory.

## Results
All results are available in the following spreadsheet:
[Google Sheets - Results](https://docs.google.com/spreadsheets/d/1ISA-CeY8QnOvP8fOFonYmIQKM-iueN_80gPgCoK5tFU/edit?usp=sharing)

The results table includes:
- **Configuration**
- **# Model Parameters**
- **AP of Action Categories**: PASS, DRIVE, HEADER, HIGH PASS, OUT, CROSS, THROW IN, SHOT, BPB, PST, FREE KICK, GOAL
- **mAP@12**: Mean AP for all classes.
- **mAP@10**: Mean AP for the first 10 classes.
  - The last two categories (GOAL and FREE KICK) are rare, so they are not considered in mAP@10.

### Best Results
- **3D ResNet-18 + LSTM + Attention**
- **X3D-M + LSTM + Attention (Stride 4)**

## References
Repository: [github.com/arturxe2/CVMasterActionRecognitionSpotting](https://github.com/arturxe2/CVMasterActionRecognitionSpotting)




