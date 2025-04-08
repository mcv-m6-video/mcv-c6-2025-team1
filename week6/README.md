<h2 align="center">WEEK 6: Task 2 - Ball Action Spotting</h2>

> [!IMPORTANT]
> This week of the project has been uploaded in a fork repository in a different place. In order to access the code and model settings, please use this command to pull the data from that repository: `git submodule update --init --recursive`.
>
> The checkpoints for our best model can be accessed from [here](https://uab-my.sharepoint.com/:f:/g/personal/1599053_uab_cat/Eus6SfAKY_RGjoS-uM5ufsUBZjXiODG4R1lJjJ2lLIqSfg?e=m0yZTM). If for any reason you don't have access to this link, please contact the administrators of this repository for permissions.

### Objective
The goal of this task is to precisely spot football ball-related actions in time, identifying the exact moment when an action occurs in a match video.

### Baseline Overview
The baseline provided uses the following architecture:
- **Backbone:** RegNet-Y (200MF), with the last fully-connected (FC) layer removed.
- **Feature extraction:** Outputs a tensor of shape (L, D), where L is the number of frames in a clip (L=50) and D is the feature dimension.
- **Classification:** FC layer projecting D to C+1 classes (C = number of action classes + 1 for no-action).
- **Activation:** Softmax (each frame belongs to only one class).
- **Inference:** Uses Non-Maximum Suppression (NMS) to refine predictions.

### Input Preprocessing
- Frame stride: 2 (from 25 FPS to 12.5 FPS).
- Clip length: 50 frames.
- Clip overlap:
  - 90% overlap for training and validation.
  - 0% overlap for testing.

### Training Details
- Loss function: Cross-Entropy Loss with weighting (1 for negative class, 5 for action classes).
- Batch size: same as in classification task.
- Epochs: 20

### Evaluation
- Average Precision (AP) calculated using SoccerNet’s official evaluation scripts with 1-second tolerance.
- Both per-class AP and mean AP are reported.

### How to run the baseline
To train and evaluate the baseline model:
```bash
python3 main_spotting.py --model baseline
```
Make sure the selected model name corresponds to a config file in the `config/` folder.

Before training:
- Generate and store clips (run at least once with `mode=store`).
- Afterward, use `mode=load` for subsequent runs.

---

## Our Experiments
We experimented with a variety of architectures and hyperparameter settings to improve upon the baseline. Our tests are listed below:

### Model Variants Tested
- **Transformer-based models**
  - x3D_M + Positional Encoding + Transformer (various configs: heads, dimensions, layers)
  - Variants with/without Gaussian blur
  - Experiments with image resolution changes (1.5x, 2x gray)
  - Clip length variations (25, 75, 100)
  - Stride variations (3, 4, 5)
  - Enhanced Positional Encoding (standard vs. improved version)
- **LSTM + Attention models**
  - x3D_M + LSTM + Attention (various heads: 2, 4, 8)
  - Use of dropout and label smoothing
  - Focal Loss (alpha=0.4, gamma=1.2)
  - Stacked LSTM layers (LSTM(2))

All results for these configurations can be found in the following spreadsheet:
[Results Excel](https://docs.google.com/spreadsheets/d/1QS-sGbw08kukQGOTI67uzOnu0Vh0EJ7M5wPVATYoF_s/edit?usp=sharing)

The spreadsheet includes:
- Configuration details
- mAP values per class
- Final mAP scores

---

## Best Results
The best performing models were:

- **x3D_M + Positional Encoding + Transformer(H=8, DIM=2048, L=1) + Img1.5x + Stride 3**  

- **x3D_M + Positional Encoding + Transformer(H=12, DIM=2048, L=1) + Img1.5x + Stride 3**  



