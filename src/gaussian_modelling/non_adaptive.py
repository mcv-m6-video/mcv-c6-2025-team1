import cv2
import numpy as np


class GaussianModelling:
    def __init__(self, alpha: float=0.01):
        """Initializes the Gaussian Modelling algorithm

        Args:
            alpha (float): The alpha parameter. Defaults to 0.01.
        """
        self.alpha = alpha
        
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
        self.mean = np.mean(frames, axis=0)
        self.variance = np.var(frames, axis=0)
        return self.mean, self.variance
    
    def get_mask(self, frame: np.ndarray):
        """Apply the Gaussian Modelling algorithm to the frame

        Args:
            frame (np.ndarray): The frame to apply the algorithm to (BGR)
        """
        assert self.mean is not None and self.variance is not None, "Background model not initialized"
        
        # Convert to grayscale
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding: if the difference between the pixel and the mean is greater than alpha * (sqrt(variance) + 2) 
        # then the pixel is foreground (255)
        mask = np.where(np.abs(gray_img - self.mean) >= self.alpha * (np.sqrt(self.variance) + 2), 255, 0)
        return mask.astype(np.uint8)
    
    def get_bounding_box(self, mask: np.ndarray, output_frame: np.ndarray, area_threshold: float=100):
        """Get the bounding box of the mask

        Args:
            mask (np.ndarray): The mask to calculate the bounding box of
            output_frame (np.ndarray): The frame to draw the bounding box on
            area_threshold (float): The minimum area of the bounding box. Defaults to 100.

        Returns:
            tuple: Tuple containing the top-left and bottom-right coordinates of the bounding box, and the output frame
        """
        # TODO: Apply morphological operations to remove noise
        
        # Get connected components
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
        
        coords = []
        for i in range(1, n_labels):
            x, y, w, h, area = stats[i]
            
            # Skip if the area is less than the threshold
            if area < area_threshold:
                continue
            
            # Get the bounding box of the largest connected component
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            coords.append((top_left, bottom_right))
            
            # Draw the bounding box on the output frame
            cv2.rectangle(output_frame, top_left, bottom_right, (0, 0, 255), 2)
            
        return coords, output_frame
        
        