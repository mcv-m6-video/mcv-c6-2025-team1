class Detection:
    """
    A class to store object detection information including bounding box,
    confidence score, camera ID, and feature vectors.
    """

    def __init__(self, frame_num, track_id, x, y, width, height, confidence, timestamp = -1.0, camera_id=None):
        """
        Initialize a Detection object.

        Args:
            frame_num (int): The frame number where detection occurs
            track_id (int): Object tracking ID (-1 if not tracked)
            x (int): X-coordinate of the top-left corner of bounding box
            y (int): Y-coordinate of the top-left corner of bounding box
            width (int): Width of the bounding box
            height (int): Height of the bounding box
            confidence (float): Confidence score of the detection
            camera_id (str, optional): ID of the camera that produced this detection
        """
        self.frame_num = frame_num
        self.track_id = track_id
        self.timestamp = timestamp

        # For bounding box
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        # For confidence and camera ID
        self.confidence = confidence
        self.camera_id = camera_id
        self.features = []

    def add_feature(self, feature_vector):
        """
        Add a feature vector to this detection.

        Args:
            feature_vector: Feature vector associated with the detection
        """
        self.features.append(feature_vector)

    @classmethod
    def from_string(cls, detection_str, timestamp=-1.0, camera_id=None):
        """
        Create a Detection object from a string representation.

        Args:
            timestamp: Timestamp of the detection
            detection_str (str): String with detection parameters (comma-separated)
            camera_id (str, optional): ID of the camera

        Returns:
            Detection: A new Detection object
        """
        parts = detection_str.strip().split(',')
        frame_num = int(parts[0])
        track_id = int(parts[1])
        x = int(parts[2])
        y = int(parts[3])
        width = int(parts[4])
        height = int(parts[5])
        confidence = float(parts[6])

        return cls(frame_num, track_id, x, y, width, height, confidence, timestamp, camera_id)

    def get_bbox(self):
        """
        Returns the bounding box coordinates as a tuple (x, y, width, height).
        """
        return self.x, self.y, self.width, self.height

    def __str__(self):
        """
        String representation of the detection.
        """
        return f"Detection(frame={self.frame_num}, id={self.track_id}, bbox=({self.x},{self.y},{self.width},{self.height}), " \
               f"conf={self.confidence:.4f}, camera={self.camera_id}, features_count={len(self.features)})"