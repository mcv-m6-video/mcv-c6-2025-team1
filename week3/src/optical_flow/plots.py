import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib.colors import Normalize


def hsv_plot(flow: np.ndarray, filename: str): # Team 1 2024 
    """Computes the HSV plot given an opticalflow into a file given a filename.

    Args:
        flow (np.ndarray): Optical flow matrix.
        filename (str): Output file name.
    """
    w, h, _ = flow.shape
    hsv = np.zeros((w, h, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    output_path = f'./results/magnitude_{filename}.png'
    cv2.imwrite(output_path, rgb)


def visualize_arrow(im1_path: str, flow: np.ndarray, filename: str): # Team 6 2024:)
    """Visualize arrows in an image.

    Args:
        im1_path (str): Input image path.
        flow (np.ndarray): Optical flow matrix
        filename (str): Output file name
    """
    im1 = cv2.imread(im1_path)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    h, w = flow.shape[:2]
    flow_horizontal = flow[:, :, 0]
    flow_vertical = flow[:, :, 1]

    step_size = 12
    X, Y = np.meshgrid(np.arange(0, w, step_size), np.arange(0, h, step_size))
    U = flow_horizontal[Y, X]
    V = flow_vertical[Y, X]
    
    magnitude = np.sqrt(U**2 + V**2)
    norm = Normalize()
    norm.autoscale(magnitude)
    cmap = cm.inferno
    
    
    # Create the plot
    plt.figure(figsize=(10, 10))
    plt.imshow(im1)
    # Do not use imshow, just create the flow vectors plot
    plt.quiver(X, Y, U, V, norm(magnitude), angles='xy', scale_units='xy', scale=1, cmap=cmap, width=0.0015)
    
    # Remove axis labels and ticks
    plt.axis('off')
    
    # Save the image directly to a file (without showing it)
    output_path = f'./results/arrow_{filename}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()