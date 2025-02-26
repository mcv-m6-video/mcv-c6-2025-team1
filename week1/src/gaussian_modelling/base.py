import cv2
import numpy as np
import jax.numpy as jnp
import os
import argparse


class GaussianModelling:
    def __init__(self, alpha: float=0.01, use_median: bool=False):
        """Initializes the Gaussian Modelling algorithm

        Args:
            alpha (float): The alpha parameter. Defaults to 0.01.
            use_median (bool): Whether to use the median instead of the mean. Defaults to False.
        """
        self.alpha = alpha
        self.use_median = use_median
        
        # Background model
        self.mean = None
        self.variance = None

    def get_bg_model(self, frames: np.ndarray):
        """Calculate the background model using the Gaussian Modelling algorithm

        Args:
            frames (np.ndarray): Numpy array of frames to calculate the background model

        Returns:
            tuple: Tuple containing the mean and variance of the background model
        """
        try:
            if self.use_median:
                self.mean = jnp.median(frames, axis=0)
            else:
                self.mean = jnp.mean(frames, axis=0)
            self.variance = jnp.var(frames, axis=0)
        except RuntimeError:
            # Just in case CUDA is not available
            if self.use_median:
                self.mean = np.median(frames, axis=0)
            else:
                self.mean = np.mean(frames, axis=0)
            self.variance = np.var(frames, axis=0)
        return self.mean, self.variance
    
    def get_mask(self, frame: np.ndarray, opening_size=5, closing_size=5):
        """Apply the Gaussian Modelling algorithm to the frame

        Args:
            frame (np.ndarray): The frame to apply the algorithm to (BGR)
        """
        assert self.mean is not None and self.variance is not None, "Background model not initialized"
        
        # Convert to grayscale
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding: if the difference between the pixel and the mean is greater than alpha * (sqrt(variance) + 2) 
        # then the pixel is foreground (255)
        mask = np.where(np.abs(gray_img - self.mean) >= self.alpha * (np.sqrt(self.variance) + 2), 255, 0).astype(np.uint8)
        
        # Remove censoring black boxes
        # mask = np.where((gray_img < 10) & (mask == 255), 0, mask).astype(np.uint8)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((opening_size, opening_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((closing_size, closing_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def get_bounding_box(self, mask: np.ndarray, output_frame: np.ndarray, area_threshold: float=100, aspect_ratio_threshold: float=1.0):
        """Get the bounding box of the mask

        Args:
            mask (np.ndarray): The mask to calculate the bounding box of
            output_frame (np.ndarray): The frame to draw the bounding box on
            area_threshold (float): The minimum area of the bounding box. Defaults to 100.
            aspect_ratio_threshold (float): The maximum aspect ratio 

        Returns:
            tuple: Tuple containing the top-left and bottom-right coordinates of the bounding box, and the output frame
        """
        # Get connected components
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        
        coords = []
        for i in range(1, n_labels):
            x, y, w, h, area = stats[i]
            
            if area < area_threshold:
                continue
            
            aspect_ratio = h/w
            if aspect_ratio > aspect_ratio_threshold:
                continue
            
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            coords.append((top_left, bottom_right))

        for (top_left, bottom_right) in coords:
            cv2.rectangle(output_frame, top_left, bottom_right, (0, 0, 255), 2)

        return coords, output_frame
    
    
## TESTS


# Function to display and test pixel stats
def test_background_model(video_path, bg_percentage=0.25):
    # Read the video
    cap = cv2.VideoCapture(video_path)
    
    # Get the total number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_bg_frames = int(bg_percentage * num_frames)
    print(f"Total frames: {num_frames}, Using {bg_percentage*100}% for background model ({n_bg_frames} frames)")

    # Initialize the background model
    gaussian_modelling = GaussianModelling(alpha=0.01, use_median=False)
    gaussian_modelling_median = GaussianModelling(alpha=0.01, use_median=True)
    bg_frames = []

    # Process the background frames
    frame_number = 0
    while frame_number <= n_bg_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_number == 0:
            # Save the first frame as the example frame
            example_frame = frame
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bg_frames.append(np.array(gray_frame))
        frame_number += 1

    # Compute the background model
    mean, variance = gaussian_modelling.get_bg_model(np.array(bg_frames))
    median, _ = gaussian_modelling_median.get_bg_model(np.array(bg_frames))
    
    # Calculate the central pixel coordinates
    height, width = example_frame.shape[:2]
    center_y, center_x = height // 2, width // 2

    # Extract the mean, median, and variance values of the central pixel
    pixel_mean = mean[center_y, center_x]
    pixel_median = median[center_y, center_x]
    pixel_variance = variance[center_y, center_x]

    # Get the mean, median, and variance for the central pixel
    print(f"Mean of the central pixel: {pixel_mean}")
    print(f"Median of the central pixel: {pixel_median}")
    print(f"Variance of the central pixel: {pixel_variance}")
    
    # Draw the pixel in red at the center of the image
    cv2.circle(example_frame, (width // 2, height // 2), 5, (0, 0, 255), -1)  # Draw a red circle
    
    # Create a folder called 'tests' if it doesn't exist
    if not os.path.exists('tests'):
        os.makedirs('tests')

    # Save the example image with central pixel highlighted
    example_image_path = os.path.join('tests', 'example_frame_pixel.jpg')
    cv2.imwrite(example_image_path, example_frame)

    cap.release()



# Function to test the mask and display the mask with and without morphological operations
def test_mask(video_path, frame_number=1000, opening_size=5, closing_size=5, alpha=0.01, use_median=False):
    # Initialize the Gaussian Modelling
    gaussian_modelling = GaussianModelling(alpha=alpha, use_median=use_median)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Read all frames until frame 1000
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    # Close the video file after reading the frames
    cap.release()
    
    # Ensure the frame number is valid
    if frame_number >= len(frames):
        print(f"Error: The video only has {len(frames)} frames.")
        return
    
    # Get the frames for background model (25% of the total frames)
    num_frames = int(0.25 * len(frames))
    bg_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames[:num_frames]]
    gaussian_modelling.get_bg_model(np.array(bg_frames))
    
    # Get the frame at frame_number (frame 1000 in this case)
    frame = frames[frame_number]
    
    # Get the mask without morphological operations
    mask_no_morph = gaussian_modelling.get_mask(frame, opening_size=0, closing_size=0)
    
    # Get the mask with morphological operations
    mask_with_morph = gaussian_modelling.get_mask(frame, opening_size=opening_size, closing_size=closing_size)
    
    # Convert the masks to BGR for visualization
    mask_no_morph_bgr = cv2.cvtColor(mask_no_morph, cv2.COLOR_GRAY2BGR)
    mask_with_morph_bgr = cv2.cvtColor(mask_with_morph, cv2.COLOR_GRAY2BGR)
    
    # Save the masks as images
    mask_no_morph_path = 'tests/mask_no_morph.jpg'
    mask_with_morph_path = 'tests/mask_with_morph.jpg'
    
    cv2.imwrite(mask_no_morph_path, mask_no_morph)
    cv2.imwrite(mask_with_morph_path, mask_with_morph)
    
    return frame, mask_no_morph_bgr, mask_with_morph_bgr

def test_bounding_box(video_path, frame_number=730, opening_size=7, closing_size=7, alpha=3.5, use_median=False, area_threshold=959, aspect_ratio_threshold=2.11):
    # Initialize the Gaussian Modelling
    gaussian_modelling = GaussianModelling(alpha=alpha, use_median=use_median)
    
    frame, mask_no_morph_bgr, mask_with_morph_bgr = test_mask(video_path, frame_number, opening_size, closing_size, alpha, use_median)
    
    mask_with_morph_gray = cv2.cvtColor(mask_with_morph_bgr, cv2.COLOR_BGR2GRAY)
    
    bounding_box_with_morph, output_frame_with_morph = gaussian_modelling.get_bounding_box(
                mask_with_morph_gray, 
                frame, 
                area_threshold=area_threshold,
                aspect_ratio_threshold=aspect_ratio_threshold
            )
    
    # Save the masks as images
    output_with_morph_path = 'tests/bounding_box_output_with_morph.jpg'
    
    cv2.imwrite(output_with_morph_path, output_frame_with_morph)


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", help="Test mode: 'background', 'mask', or 'bounding_box'", required=True)
    args = parser.parse_args()

    video_path = '/ghome/c3mcv02/mcv-c6-2025-team1/data/AICity_data/train/S03/c010/vdo.avi'

    if args.test == "background":
        test_background_model(video_path)
    elif args.test == "mask":
        test_mask(video_path, frame_number=730, opening_size=7, closing_size=7, alpha=3.5, use_median=False)
    elif args.test == "bounding_box":
        test_bounding_box(video_path, frame_number=730, opening_size=7, closing_size=7, alpha=3.5, use_median=False, area_threshold=918, aspect_ratio_threshold=2.11)
    else:
        print("Invalid test mode. Use 'background', 'mask', or 'bounding_box'.")

