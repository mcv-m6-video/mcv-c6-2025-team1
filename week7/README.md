<h2 align="center">WEEK 6 + 7: Ball Action Spotting – Temporal Aggregation Exploration</h2>

> [!IMPORTANT]
> This week’s experiments build on the same codebase as the previous week, located in a forked repository. Please make sure to initialize the repository correctly using:
> 
> `git submodule update --init --recursive`
>
> The checkpoints for our best models are available here: [Download Checkpoints](https://uab-my.sharepoint.com/:f:/g/personal/1599053_uab_cat/Eus6SfAKY_RGjoS-uM5ufsUBZjXiODG4R1lJjJ2lLIqSfg?e=m0yZTM)

### Objective
This task focuses on evaluating the effect of **temporal aggregation** on spotting performance. The objective remains to precisely spot football ball-related actions in time, but this week we explore how downsampling, multi-scale approaches, and displacement heads affect model accuracy.

### Reference Approaches
Inspired by recent literature:
- Shi, Dingfeng, et al. "Tridet: Temporal action detection with relative boundary modeling." CVPR 2023.
- Xarles, Artur, et al. "T-DEED: Temporal-Discriminability Enhancer Encoder-Decoder for Precise Event Spotting in Sports Videos." CVPR 2024.

We experimented with different architectural choices for temporal modeling and displacement-based event localization.

### Baseline Recap
The baseline from the previous week includes:
- **Backbone:** RegNet-Y (200MF), FC layer removed
- **Feature Extraction:** Outputs (L, D) with L=50 frames, D=feature dimension
- **Classifier:** FC layer mapping D to C+1 classes
- **Activation:** Softmax
- **Inference:** Non-Maximum Suppression (NMS)
- **Input Preprocessing:**
  - Frame stride: 2 (reduces 25 FPS to 12.5 FPS)
  - Clip length: 50 frames
  - Overlap: 90% for training/validation, 0% for testing
- **Loss Function:** Cross-Entropy with class weighting (1 for background, 5 for action classes)
- **Epochs:** 20

### Our Experiments

This week, we tested a wide variety of architectures and hyperparameters focusing on **temporal resolution** and **aggregation strategy**.

#### T-DEED-based Models

- T-DEED(L=1) + X3D_M + No Displacement
- T-DEED(L=2) + X3D_M + No Displacement
- T-DEED(L=2, KS = 5, R =2) + X3D_M + No Displacement
- T-DEED(L=2, KS = 5, R =2) + X3D_M + No Displacement + ImgSize2x**(Gray)
- T-DEED(L=2, KS = 5, R =2) + X3D_M + No Displacement + Stride 1
- T-DEED(L=2, KS = 5, R =2) + X3D_M + No Displacement + Stride 3
- T-DEED(L=2, KS = 5, R =2) + X3D_M + No Displacement + Stride 3 + ImgSize1.5x*
- T-DEED(L=2, KS = 5, R =2) + X3D_M + No Displacement + Stride 3 + ImgSize2x**(Gray)
- T-DEED(L=2, KS = 5, R =2) + X3D_M + No Displacement + Stride 4
- T-DEED(L=2, KS = 5, R =2) + X3D_M + No Displacement + 50% Overlap
- T-DEED(L=2, KS = 3, R =2) + X3D_M + No Displacement
- T-DEED(L=2) + X3D_M + No Displacement + Mixup
- T-DEED(L=2) + X3D_M + No Displacement + ImgSize1.5x*
- T-DEED(L=2, KS=5, R=2) + X3D_M + No Displacement + ImgSize1.5x*
- T-DEED(L=2) + X3D_M + No Displacement + ClipLen25
- T-DEED(L=2) + X3D_M + No Displacement + ClipLen100
- T-DEED(L=2) + X3D_M + Displacement(r=2)
- T-DEED(L=2) + X3D_M + Displacement(r=4)
- T-DEED(L=2) + X3D_M + Displacement(r=4) + ImgSize1.5x
- T-DEED(L=4) + X3D_M + No Displacement
- T-DEED(L=4, KS=5, R=2) + X3D_M + No Displacement
- T-DEED(L=4, KS=3, R=2) + X3D_M + No Displacement
- T-DEED(L=2) + X3D_L + No Displacement
- T-DEED + No Displacement
- T-DEED + Displacement(r=4)
- T-DEED(RegNet-Y 800) + Displacement(r=4)
- T-DEED + Displacement(r=4) + Resolution(x1.5)
- T-DEED + Displacement(r=4) + ClipLen75

#### TCN-based Models

- x3D_M + TCN(C=64, L=1, K=4) + No Displacement
- x3D_M + TCN(C=64, L=2, K=3) + No Displacement
- x3D_M + TCN(C=64, L=2, K=4) + No Displacement
- x3D_M + TCN(C=32, L=2, K=5) + No Displacement
- x3D_M + TCN(C=64, L=2, K=5) + No Displacement
- x3D_M + TCN(C=64, L=2, K=5) + No Displacement + ImgSize2x**(Gray)
- x3D_M + TCN(C=64, L=2, K=5) + No Displacement + Stride 1
- x3D_M + TCN(C=64, L=2, K=5) + No Displacement + Stride 3
- x3D_M + TCN(C=64, L=2, K=5) + No Displacement + ImgSize1.5x*
- x3D_M + TCN(C=128, L=2, K=5) + No Displacement
- x3D_M + TCN(C=64, L=2, K=6) + No Displacement
- x3D_M + TCN(C=64, L=3, K=4) + No Displacement

### Evaluation
Evaluation is conducted using SoccerNet’s official script with 1-second tolerance.
- Metrics reported: Average Precision (AP) per class and mean AP (mAP)
- Results are collected in the spreadsheet below

[Results Spreadsheet](https://docs.google.com/spreadsheets/d/18jn-YjT2xS1efukN7Oj5RSKX9svhdT5q-8vW6ecafg0/edit?usp=sharing)

### Best Results
The best performing configuration was:

**T-DEED (L=2, KS=5, R=2) + X3D_M + No Displacement + Stride 3 + ImgSize 2x (Grayscale)**

This configuration achieved the highest overall mAP across all ball-action classes.

### How to Run
To train and evaluate a model:
```bash
python3 main_spotting.py --model <model_name>
```


