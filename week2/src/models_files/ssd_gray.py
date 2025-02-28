from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from PIL import Image

import numpy as np 
import torch

import cv2


class SSD:
    def __init__(self, box_score_thresh: float=0.9):
        """SSD model using VGG-16 backbone.

        Args:
            box_score_thresh (float, optional): Threshold for box score. Defaults to 0.9.
        """
        self.box_score_thresh = box_score_thresh
        
        # Load weights default is COCO_V1
        self.weights = SSD300_VGG16_Weights.DEFAULT
        self.preprocess = self.weights.transforms()
        
        # Set the model to evaluation mode
        self.model = ssd300_vgg16(weights=self.weights, score_thresh=self.box_score_thresh)
        self.model.eval()
    
    def decode_img(self, image: np.ndarray) -> torch.Tensor:
        """Convert image from OpenCV to Tensor.

        Args:
            image (np.ndarray): Image in grayscale.

        Returns:
            torch.Tensor: Image in tensor format.
        """
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        return image

    def predict(self, image: np.ndarray) -> torch.Tensor:
        """Predict bounding boxes on an image.

        Args:
            image (np.ndarray): NumPy array of image in grayscale.

        Returns:
            torch.Tensor: Predictions from the model.
        """
        #print("Predicting image with shape:", image.shape)
        
        # Convert image to tensor
        image = self.decode_img(image)
        
        #print("Image shape after unsqueeze:", image.shape)
        
        batch = [self.preprocess(image)]
        predictions = self.model(batch)
        
        #print("Predictions shape:", predictions[0]['boxes'].shape)
        #print("Predictions:", predictions)
        
        return predictions

    def draw_predictions(self, image: torch.Tensor | np.ndarray, prediction: torch.Tensor) -> Image:
        """Draw bounding boxes on an image.

        Args:
            image (torch.Tensor | np.ndarray): Image to draw bounding boxes on.
            prediction (torch.Tensor): Prediction for that frame from the model.

        Returns:
            Image: Image with bounding boxes drawn.
        """
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        #print("Image shape before drawing:", image.shape)
        
        labels = [self.weights.meta["categories"][i] for i in prediction["labels"]]
        
        image = draw_bounding_boxes(image, prediction['boxes'], 
                                    labels=labels, 
                                    colors="red", 
                                    width=4, font="DejaVuSans",
                                    font_size=30)
        image = to_pil_image(image.detach())
        return image
    
    def get_classes(self) -> list:
        """Get the classes of the model.

        Returns:
            list: List of classes in the model.
        """
        return self.weights.meta["categories"]


# Example usage
if __name__ == "__main__":
    # Let's load the video and get the first frame
    video = cv2.VideoCapture("/ghome/c3mcv02/mcv-c6-2025-team1/data/AICity_data/train/S03/c010/vdo.avi")
    ret, frame = video.read()
    
    # Initialize the FasterRCNN model
    model = SSD(box_score_thresh=0.5)
    
    # Predict bounding boxes on the first frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    predictions = model.predict(frame)
    
    # Draw bounding boxes on the first frame
    image = model.draw_predictions(frame, predictions[0])
    
    # Save the image with bounding boxes
    image.save("ssd_fist_frame.jpg")