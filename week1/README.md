<h2 align="center">WEEK 1: Tasks</h2>

## Table of Contents

- [Project Structure W1](#project-structure-w1)
- [Task 1.1: Gaussian Modelling](#task-11-gaussian-modelling)
- [Task 1.2: mAP@0.5 vs Alpha](#task-12-map05-vs-alpha)
- [Task 2.1: Adaptive Modelling](#task-21-adaptive-modelling)
- [Task 2.2: Comparison of Adaptive vs. Non-Adaptive](#task-22-comparison-of-adaptive-vs-non-adaptive)
- [Task 3: Comparison with State-of-the-Art](#task-3-comparison-with-state-of-the-art)


### Project Structure W1

    week1/
    ├── src/
    │   ├── gaussian_modelling/
    │   │   ├── adaptive.py
    │   │   └── base.py
    │   ├── main_adaptive.py
    │   ├── main_optuna.py
    │   ├── main.py
    │   └── utils.py
    
### Task 1.1: Gaussian modelling

#### Overview

This task implements a Gaussian-based background modeling algorithm to segment the foreground in a video sequence. The method follows these steps:

1. **Model the background**  
   - The first 25% of the frames are used to estimate the background.
   - The background model is computed using either the mean or the median of the pixel values.
   - The variance of each pixel is also computed.

2. **Segment the foreground**  
   - The remaining 75% of the frames are used for foreground segmentation.
   - A pixel is classified as foreground if its intensity significantly deviates from the background model based on a threshold.

3. **Post-processing**  
   - Morphological operations (opening and closing) are applied to reduce noise.
   - Bounding boxes are computed to detect objects.

#### Implementation

The Gaussian Modelling is implemented in the `GaussianModelling` class, with the following functions:

- **Background Model Calculation:**

The function `def get_bg_model(self, frames: np.ndarray)` computes the mean (or median) and variance for each pixel. In the main function, the parameter `frames` corresponds to the first 25% of the frames of the video.

By default, the function uses **JAX (`jnp`)** to take advantage of GPU acceleration with CUDA. If CUDA is not available, it falls back to **NumPy (`np`)** to ensure compatibility.

***Example:***

To test the background model function, run the following command:

```bash
cd src/gaussian_modelling/
python3 base.py --test background
```

This will process the video frames and calculate the background model for the central pixel. The output should look like this:

```bash
Total frames: 2141, Using 25.0% for background model (535 frames)
Mean of the central pixel: 172.02985074626866
Median of the central pixel: 173.0
Variance of the central pixel: 19.898362664290488
```

Here is an image showing the first frame of the video with the central pixel highlighted in red:

![example_frame_pixel](https://github.com/user-attachments/assets/29274e0d-3a4f-48d2-8382-8ab0ef5c65aa)


- **Foreground Segmentation:**

The function `def get_mask(self, frame: np.ndarray, opening_size=5, closing_size=5)` segments the foreground by comparing each pixel to the background model.

1. **Convert to Grayscale:**  
   Since the algorithm operates on intensity values, the input frame (BGR) is converted to grayscale.

2. **Thresholding:**
   A pixel is classified as foreground if its intensity deviates significantly from the background model. The threshold is determined as:

   $| I(x, y) - \mu(x, y) | \geq \alpha \cdot (\sqrt{\sigma^2(x, y)} + 2)$

   where $I(x, y)$ is the pixel intensity, $\mu(x, y)$ and $\sigma^2(x, y)$ are the mean (or median) and variance of the background model, and $\alpha$ is a tunable parameter. The pixel will be classified as **foreground** if the equation is satisfied, and as **background** if it is not.
   
4. **Morphological Post-Processing:**  
   To reduce noise, **morphological opening** is applied, followed by **morphological closing** to refine object boundaries. The kernel sizes for these operations are adjustable via `opening_size` and `closing_size`.

***Example:***

To test the mask function, run the following command:

```bash
python3 base.py --test mask
```

This test compares the masks generated with and without morphological operations (opening and closing). The test uses frame 730 from the video with the following parameters: `opening_size=7`, `closing_size=7`, `alpha=3.5`, and `use_median=False`. The masks are saved as `mask_no_morph.jpg` (without operations) and `mask_with_morph.jpg` (with opening and closing operations). The morphological operations refine object boundaries and reduce noise, improving the quality of the mask.

**Results:**

| **Without Morphological Operations** | **With Morphological Operations**  |
|--------------------------------------|------------------------------------|
| ![mask_no_morph](https://github.com/user-attachments/assets/5fa62d40-4b5d-4ef1-9f82-dbbf9741976f) | ![mask_with_morph](https://github.com/user-attachments/assets/d4c0413b-3915-4b4a-9329-1dea785841bf) |


- **Bounding Box Calculation**

The function `def get_bounding_box(self, mask: np.ndarray, output_frame: np.ndarray, area_threshold: float=100, aspect_ratio_threshold: float=1.0)` detects objects in the segmented mask and draws bounding boxes around them.

1. **Connected Components Analysis:**  
   The function uses `cv2.connectedComponentsWithStats` to identify separate objects in the mask.

2. **Filtering by Area:**  
   - Small components with an area below `area_threshold` are ignored.

3. **Filtering by Aspect Ratio:**  
   - Objects with an aspect ratio (height/width) greater than `aspect_ratio_threshold` are discarded.  
   - This helps filter out elongated objects like bicycles, ensuring that only cars are detected.

4. **Drawing Bounding Boxes:**  
   - The remaining objects are enclosed in red bounding boxes and drawn on the `output_frame`.

***Example:***

To test the mask function, run the following command:

```bash
python3 base.py --test bounding_box
```

This test processes frame 730 of the video and applies bounding boxes on objects detected in the mask. The bounding boxes, which are **red**, will be drawn only on objects that pass the area and aspect ratio filters.

For testing we use diferent parameters for `area_threshold` and `aspect_ratio_threshold`:

| **area_threshold=300 & aspect_ratio_threshold=1.5** | **area_threshold=918 & aspect_ratio_threshold=2.11** |
|-----------------------------------------------------|-------------------------------------------------------|
| ![bounding_box_output_with_morph_1](https://github.com/user-attachments/assets/9ac40d07-4b4d-45ea-9c74-0c2e51c592a8) | ![bounding_box_output_with_morph](https://github.com/user-attachments/assets/a548a7c0-ee6a-4fbd-bd60-6ba6dd5683e8) |


### Task 1.2: mAP0.5 vs Alpha


```bash
cd src/
python3 main.py -v=/ghome/c3mcv02/mcv-c6-2025-team1/data/AICity_data/train/S03/c010/vdo.avi -a=3.5 -t=918 --annotations=/ghome/c3mcv02/mcv-c6-2025-team1/data/ai_challenge_s03_c010-full_annotation.xml --use_median --opening_size=3 --closing_size=13 -r=2.11 -o="output_readme.avi" -m="mask_readme.avi"
```

Mean Average Precision: 0.43533866925279


### Task 2.1: Adaptive modelling

```bash
cd src/
python3 main_adaptive.py -v=/ghome/c3mcv02/mcv-c6-2025-team1/data/AICity_data/train/S03/c010/vdo.avi -a=2.5 -o=output_readme_adaptive.avi -m=mask_readme_adaptive.avi -rho=0.01  -t=959 --annotations=/ghome/c3mcv02/mcv-c6-2025-team1/data/ai_challenge_s03_c010-full_annotation.xml --use_median --opening_size=3 --closing_size=13 -r=1.2
```

Mean Average Precision: 0.7226676165709937

### Task 2.2: Comparison of adaptive vs. non-adaptive

### Task 3: Comparison with state-of-the-art

```bash
python3 -m src.stateofart.methods
```
opening_size=3 closing_size=13

#### ZBS (Zero-shot Background Substraction) method
To complete the ZBS part, please follow the installation and usage instructions from the [official ZBS repository](https://github.com/CASIA-IVA-Lab/ZBS). Extract the masks for each frame using FG threshold of 0.4 and move threshold of 0.8, with confidence 0.6, altough you may use different setting.

For evaluation, use this command which will output the mAP for the masks predicted by ZBS:

```bash
python3 -m src.stateofart.eval_zbs -m <ZBS_MASK>.avi -gt <GT_ANNOTATIONS>.xml -v
```

For further details on other command options use:

```bash
python3 -m src.stateofart.eval_zbs --help
```




