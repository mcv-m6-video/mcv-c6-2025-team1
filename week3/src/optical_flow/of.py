import numpy as np
import cv2
from utils import *
import pyflow
import matplotlib.pyplot as plt
import time
from plots import *

# Load images
img1 = cv2.imread("/ghome/c5mcv01/mcv-c6-2025-team1/data_w3/data_stereo_flow/training/image_0/000045_10.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("/ghome/c5mcv01/mcv-c6-2025-team1/data_w3/data_stereo_flow/training/image_0/000045_11.png",cv2.IMREAD_GRAYSCALE)
gt_path = "/ghome/c5mcv01/mcv-c6-2025-team1/data_w3/data_stereo_flow/training/flow_noc/000045_10.png"
gt = read_png_file(gt_path)


img1 = np.atleast_3d(img1.astype(float) / 255.0)
img2 = np.atleast_3d(img2.astype(float) / 255.0)

# Optical flow parameters
alpha = 0.012  # Smoothness
ratio = 0.75  # Downscale ratio
minWidth = 20  # Min width of output flow
nOuterFPIterations = 1
nInnerFPIterations = 1
nSORIterations = 30
colType = 1  

start_time = time.time()
# Compute flow
u,v,_ = pyflow.coarse2fine_flow(
    img1, img2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType
)
inference_time = time.time() - start_time

flow_pyflow = np.dstack((u, v))
    
msen_pyflow = calculate_msen(gt,flow_pyflow)
pepn_pyflow = calculate_pepn(gt, flow_pyflow)

print(f"\nMSEN (PyFlow): {msen_pyflow:.4f}")
print(f"\nPEPN (PyFlow): {pepn_pyflow * 100:.2f}%")

pyflow_hsv = hsv_plot(flow_pyflow)
gt_hsv = hsv_plot(gt)
    
# Save optical flow result
cv2.imwrite("optical_flow_result.png", pyflow_hsv)
cv2.imwrite("optical_flow_gt.png", gt_hsv)

visualize_arrow(im1_path, gt, filename="GT")
visualize_arrow(im1_path, flow_pyflow, filename="PyFlow")

