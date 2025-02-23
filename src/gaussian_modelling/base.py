import cv2
import numpy as np
import jax.numpy as jnp


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