import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib.colors import Normalize


def hsv_plot(flow: np.ndarray, filename: str):
    """Computes the HSV plot given an opticalflow into a file given a filename.

    Args:
        flow (np.ndarray): Optical flow matrix.
        filename (str): Output file name.
    """
    results_dir = './results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # The shape should be h, w, _ (height first, width second)
    h, w, _ = flow.shape  # Fixed: swapped w and h
    
    # Create HSV array with correct dimensions (h, w) not (w, h)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    
    # Ensure flow components are float32
    flow_x = flow[..., 0].astype(np.float32)
    flow_y = flow[..., 1].astype(np.float32)
    
    # Calculate magnitude and angle
    mag, ang = cv2.cartToPolar(flow_x, flow_y)
    
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    output_path = f'{results_dir}/magnitude_{filename}.png'
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


def generate_optical_flow_color_wheel(size=256):
    """Generates an HSV-based optical flow color wheel.
    
    Args:
        size (int): Size of the output image (size x size).
        
    Returns:
        np.ndarray: RGB image of the color wheel.
    """
    radius = size // 2
    y, x = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    
    # Convert cartesian coordinates to polar
    mag = np.sqrt(x**2 + y**2)
    ang = np.arctan2(y, x)  # Angle in radians
    
    # Normalize angle to [0, 180] for OpenCV HSV
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) * 180 / np.pi / 2).astype(np.uint8)  # Hue
    hsv[..., 1] = 255  # Saturation is max
    hsv[..., 2] = (cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)  # Value based on magnitude
    
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return rgb

# Save the color wheel
color_wheel = generate_optical_flow_color_wheel(256)
cv2.imwrite('optical_flow_color_wheel.png', color_wheel)
