from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from PIL import Image

import numpy as np 
import torch
import cv2
import skimage

from matplotlib import pyplot as plt
import matplotlib.patches as patches


class SSD_ResNet50:
    def __init__(self,  box_score_thresh: float=0.9):
        """SSD model using ResNet-50 backbone.

        Args:
            box_score_thresh (float, optional): Threshold for box score. Defaults to 0.9.
        """
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
        self.model.to('cuda')
        self.model.eval()
        self.box_score_thresh = box_score_thresh
    
    def decode_img(self, image: np.ndarray) -> torch.Tensor:
        """Convert image from OpenCV to Tensor.

        Args:
            image (np.ndarray): Image in grayscale.

        Returns:
            torch.Tensor: Image in tensor format.
        """
        # TODO: Decode image from OpenCV to Tensor
        image = cv2.resize(image, (300, 300))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255
        image = np.expand_dims(image, axis=0)
        image = self.utils.prepare_tensor(image)
        return image

    def predict(self, image: np.ndarray) -> torch.Tensor:
        """Predict bounding boxes on an image.

        Args:
            image (np.ndarray): NumPy array of image in grayscale.

        Returns:
            torch.Tensor: Predictions from the model.
        """
        print("Predicting image with shape:", image.shape)
        
        # Convert image to tensor
        image = self.decode_img(image)
        
        with torch.no_grad():
            detections_batch = self.model(image)
        
        results_per_input = self.utils.decode_results(detections_batch)
        best_results_per_input = [self.utils.pick_best(results, self.box_score_thresh) for results in results_per_input]
        # 3 and 8 (car and truck)
        classes_to_labels = self.utils.get_coco_object_dictionary()
        
        return best_results_per_input

    def draw_predictions(self, image: torch.Tensor | np.ndarray, BB, confidences) -> Image:
        """Draw bounding boxes on an image.

        Args:
            image (torch.Tensor | np.ndarray): Image to draw bounding boxes on.
            prediction (torch.Tensor): Prediction for that frame from the model.

        Returns:
            Image: Image with bounding boxes drawn.
        """

        # Convert the image to BGR format for OpenCV
        height, width, channels = image.shape

        # Loop through the bounding boxes and draw them
        for idx in range(len(BB)):
            left, bot, right, top = BB[idx]
            x, y, w, h = [val for val in [left, bot, (right - left), (top - bot)]]
            x = int(x*width)
            y = int(y*height)
            w = int(w*width)
            h = int(h*height)
            print(f"x: {x}, y: {y}, w: {w}, h: {h}")

            # Draw the rectangle
            cv2.rectangle(image, (x, y), ((x + w), (y + h)), (0, 0, 255), 2)  # Red color, thickness 2

            # Add text (class label and confidence) to the bounding box
            label = f"{confidences[idx]*100:.0f}%"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Save the image with detections as a new file
        output_image_path = "image_with_predictions_1.png"
        cv2.imwrite(output_image_path, image)
        print(f"Image with predictions saved as {output_image_path}")
        return image



# Example usage
if __name__ == "__main__":
    # Let's load the video and get the first frame
    video = cv2.VideoCapture("/ghome/c3mcv02/mcv-c6-2025-team1/data/AICity_data/train/S03/c010/vdo.avi")
    ret, frame = video.read()
    
    # Initialize the FasterRCNN model
    model = SSD_ResNet50(box_score_thresh=0.1)
    
    # Predict bounding boxes on the first frame
    predictions = model.predict(frame)
    print(f"Predictions: {predictions}")
    bboxes, classes, confidences = predictions[0]
    print(f"bboxes: {bboxes}, classes: {classes}, confidences: {confidences}")   
    indices = (classes == 3) | (classes == 8)
    filtered_bboxes = bboxes[indices]
    filtered_classes = classes[indices]
    filtered_confidences = confidences[indices]
    print(f"bboxes: {filtered_bboxes}, classes: {filtered_classes}, confidences: {filtered_confidences}")   

    # Draw bounding boxes on the first frame
    image = model.draw_predictions(frame, filtered_bboxes, filtered_confidences)
    


