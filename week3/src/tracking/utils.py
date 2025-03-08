import numpy as np
from lxml import etree


def mean_avg_precision(gt, pred, iou_threshold=0.5):
    """Calculate the mean average precision for a given ground truth and prediction. From Team 5-2024 (slightly modified).

    Args:
        gt (list): List of ground truth bounding boxes
        pred (list): List of predicted bounding boxes
        iou_threshold (float): The intersection over union threshold. Defaults to 0.5.

    Returns:
        float: The mean average precision
    """
    if len(gt) == 0 or len(pred) == 0:
        return 1 if len(gt) == len(pred) == 0 else 0
    
    # Initialize variables
    tp = np.zeros(len(pred))
    fp = np.zeros(len(pred))
    gt_matched = [False] * len(gt)
    
    # Loop through each prediction
    for i, p in enumerate(pred):
        ious = [iou(p, g) for g in gt]
        if len(ious) == 0:
            fp[i] = 1
            continue
        
        max_iou_idx = np.argmax(ious)
        max_iou = ious[max_iou_idx]
        
        if max_iou >= iou_threshold and not gt_matched[max_iou_idx]:
            tp[i] = 1
            gt_matched[max_iou_idx] = True
        else:
            fp[i] = 1
    
    # Calculate precision and recall
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / len(gt) # len(gt) is equivalent to TP + FN
    precision = tp / (tp + fp)
    
    # Generate graph with the 11-point interpolated precision-recall curve (Team5-2024)
    recall_interp = np.linspace(0, 1, 11)
    precision_interp = np.zeros(11)
    for i, r in enumerate(recall_interp):
        array_precision = precision[recall >= r]
        if len(array_precision) == 0:
            precision_interp[i] = 0
        else:
            precision_interp[i] = max(precision[recall >= r])
    return np.mean(precision_interp)

def iou(boxA, boxB):
    """Calculate the intersection over union (IoU) of two bounding boxes. From Team 5-2024 (slightly modified)
    
    Format of bounding boxes is top-left and bottom-right coordinates [x1, y1, x2, y2].

    Args:
        boxA (list): List containing the coordinates of the first bounding box [x1, y1, x2, y2]
        boxB (list): List containing the coordinates of the second bounding box [x1, y1, x2, y2]

    Returns:
        float: The IoU value
    """
    # Calculate the intersection area
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # Calculate the area of each bounding box
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # Calculate the union area
    unionArea = boxAArea + boxBArea - interArea
    
    # Calculate the IoU
    return interArea / unionArea


def read_annotations(annotations_path: str):
    """Read the annotations from the XML file and extract car bounding boxes.

    Args:
        annotations_path (str): Path to the XML file

    Returns:
        dict: Dictionary containing the car bounding boxes for each frame
    """
    # Read the XML file where the annotations (GT) are
    tree = etree.parse(annotations_path)
    # Take the root element of the XML file
    root = tree.getroot()
    # Where car bounding boxes will be stored
    car_boxes = {}

    # Iterate over all car annotations (label = "car") 
    for track in root.xpath(".//track[@label='car']"):
        # Iterate over all 2D bounding boxes annotated for the car
        for box in track.xpath(".//box"):
            # You can skip the "parked" or "occluded" filtering, so we don't need to check those
            frame = box.get("frame")

            # Extract the bounding box attributes
            box_attributes = {
                "xtl": float(box.get("xtl")),  # Top-left X
                "ytl": float(box.get("ytl")),  # Top-left Y
                "xbr": float(box.get("xbr")),  # Bottom-right X
                "ybr": float(box.get("ybr")),  # Bottom-right Y
                "occluded": int(box.get("occluded")),  # Occlusion status
            }

            # Store the bounding box for the given frame
            if frame in car_boxes:
                car_boxes[frame].append(box_attributes)
            else:
                car_boxes[frame] = [box_attributes]
    
    return car_boxes
