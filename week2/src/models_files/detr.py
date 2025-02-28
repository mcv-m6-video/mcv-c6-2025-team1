import torch
import cv2
import numpy as np
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.ops import box_convert
from PIL import ImageDraw, ImageFont
from PIL import Image

class DETR:
    def __init__(self, score_threshold: float = 0.9):
        """DEtection TRansformer (DETR) with ResNet50 backbone.

        Args:
            score_threshold (float, optional): Confidence threshold for displaying boxes. Defaults to 0.9.
        """
        self.score_threshold = score_threshold
        
        # Load pretrained DETR model on COCO dataset
        self.model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True, force_reload=True)
        self.model.eval()
        
        # COCO categories
        self.categories = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
            'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet',
            'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def decode_img(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess the image for DETR (BGR -> RGB, normalization, and conversion to tensor)."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, we change it to RGB
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # (H, W, C) -> (C, H, W)
        return image.unsqueeze(0)  # Add batch dimension

    def predict(self, image: np.ndarray) -> dict:
        """Make predictions on the image and filter only 'car' and 'truck'."""
        tensor_image = self.decode_img(image)
        outputs = self.model(tensor_image)

        # Extract boxes, classes, and scores
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]  # Exclude "__background__" class
        keep = probas.max(-1).values > self.score_threshold  # Filter by confidence threshold

        boxes = outputs['pred_boxes'][0, keep]  # Get bounding boxes
        labels = probas[keep].argmax(-1)  # Detected classes
        scores = probas[keep].max(-1).values  # Confidence scores
        
        # Convert boxes from (cx, cy, w, h) to (xmin, ymin, xmax, ymax)
        boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

        # Get indices of 'car' and 'truck' in the category list
        allowed_classes = {"car", "truck"}
        allowed_indices = {self.categories.index(cls) for cls in allowed_classes}

        # Filter predictions based on allowed classes
        selected = [i for i, label in enumerate(labels.tolist()) if label in allowed_indices]

        return {
            'boxes': boxes[selected],
            'labels': labels[selected],
            'scores': scores[selected]
        }
    
    def draw_predictions(self, image: np.ndarray, prediction: dict) -> Image:
        """Draw the bounding boxes on the image."""
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # Convert to tensor format (C, H, W)

        labels = [self.categories[i] for i in prediction["labels"]]
        
        # Scale boxes to pixel coordinates
        boxes = prediction['boxes'] * torch.tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        
        image_with_boxes = draw_bounding_boxes(
            image_tensor, 
            boxes=boxes,
            labels=labels, 
            colors="red", 
            width=3,
            font="DejaVuSans",
            font_size=30
        )
        return to_pil_image(image_with_boxes)


# Example usage with a video
if __name__ == "__main__":
    # Load video
    video_path = "/ghome/c3mcv02/mcv-c6-2025-team1/data/AICity_data/train/S03/c010/vdo.avi"
    video = cv2.VideoCapture(video_path)
    
    ret, frame = video.read()  # Read first frame
    
    if ret:
        model = DETR()
        predictions = model.predict(frame)  # Get predictions
        image = model.draw_predictions(frame, predictions)  # Draw boxes
        image.save("first_frame_detr.jpg")  # Save image with detections
