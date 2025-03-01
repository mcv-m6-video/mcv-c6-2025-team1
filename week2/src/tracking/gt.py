from lxml import etree

def read_annotations(annotations_path: str):
    """Read the annotations from the XML file and extract car bounding boxes.

    Args:
        annotations_path (str): Path to the XML file

    Returns:
        dict: Dictionary containing the car bounding boxes for each frame
    """
    # Read the XML file where the annotations (GT) are
    tree = etree.parse(annotations_path)
    root = tree.getroot()

    # Where car bounding boxes will be stored
    car_boxes = {}

    # Iterate over all car annotations (label = "car")
    for track in root.xpath(".//track[@label='car']"):
        object_id = int(track.get("id"))  # Get the unique track id for each object (car)
        
        
        # Iterate over all 2D bounding boxes annotated for the car
        for box in track.xpath(".//box"):
            # Extract the frame number and bounding box coordinates
            frame = int(box.get("frame"))
            xtl = float(box.get("xtl"))  # Top-left X
            ytl = float(box.get("ytl"))  # Top-left Y
            xbr = float(box.get("xbr"))  # Bottom-right X
            ybr = float(box.get("ybr"))  # Bottom-right Y

            # Calculate the width and height of the bounding box
            bbox_width = xbr - xtl
            bbox_height = ybr - ytl

            # Store the bounding box for the given frame and object
            if frame not in car_boxes:
                car_boxes[frame] = []
            car_boxes[frame].append({
                "object_id": object_id,
                "xtl": xtl,
                "ytl": ytl,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height
            })
    
    return car_boxes



def write_gt_txt(car_boxes, output_path="gt.txt"):
    """Write ground truth annotations to a text file in the required format.

    Args:
        car_boxes (dict): Dictionary containing car bounding boxes for each frame
        output_path (str): Path to the output ground truth text file
    """
    with open(output_path, 'w') as file:
        for frame_id, boxes in sorted(car_boxes.items()):
            # Add 1 to the frame_id for starting from frame 1
            frame_id += 1
            
            for box in boxes:
                # Add 1 to the object_id (track_id)
                object_id = int(box["object_id"]) 
                bb_left = box["xtl"]
                bb_top = box["ytl"]
                bb_width = box["bbox_width"]
                bb_height = box["bbox_height"]
                conf = 1  # Confidence for ground truth is always 1
                file.write(f"{frame_id}, {object_id}, {bb_left}, {bb_top}, {bb_width}, {bb_height}, {conf}, -1, -1, -1\n")

annotations_path = "/ghome/c3mcv02/mcv-c6-2025-team1/data/ai_challenge_s03_c010-full_annotation.xml"
car_boxes = read_annotations(annotations_path)
write_gt_txt(car_boxes, "gt.txt")