# utilsNeeded.py
import cv2
import numpy as np
import winsound
import hashlib


def run_yolov8_inference(model, frame):
    """
    Perform object detection on a single image using a preloaded YOLOv8 model.

    Parameters:
    - model: An instance of a YOLOv8 model ready for inference.
    - frame: An image in BGR format (numpy array) for object detection.

    Returns:
    A list of detections, each represented as a list containing:
    [bounding box coordinates (x1, y1, x2, y2), confidence score, class ID, class name]
    """
    # Perform inference with the YOLOv8 model
    results = model.predict(frame)
    detections = []

    # Assuming the first item in results contains the detection information
    if results:
        detection_result = results[0]
        xyxy = detection_result.boxes.xyxy.numpy()  # Bounding box coordinates
        confidence = detection_result.boxes.conf.numpy()  # Confidence scores
        class_ids = detection_result.boxes.cls.numpy().astype(int)  # Class IDs
        class_names = [model.model.names[cls_id] for cls_id in class_ids]  # Class names

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, xyxy[i])
            conf = confidence[i]
            cls_id = class_ids[i]
            class_name = class_names[i]
            detections.append([x1, y1, x2, y2, conf, cls_id, class_name])

    return detections
# Function to run YOLOv5 inference on a frame and extract detections
def run_yolov5_inference(model, frame):
    """
    Runs YOLOv5 inference on a frame and extracts detections.

    Args:
        model: YOLOv5 model instance.
        frame: Input frame for object detection.

    Returns:
        detections: List of detected objects, each represented as [x1, y1, x2, y2, confidence, class_id, class_name].
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    detections = []
    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = xyxy
        class_name = model.names[int(cls.item())]  # Get class name
        detections.append([x1.item(), y1.item(), x2.item(), y2.item(), conf.item(), cls.item(), class_name])
    return detections




# Function to generate unique color for each class ID
def get_color_by_id(class_id):
    """
    Generates a unique color for each class ID to ensure consistency across runs.

    Parameters:
        class_id (int): Unique identifier for the class.

    Returns:
        list: RGB color values.
    """
    hash_value = hashlib.sha256(str(class_id).encode()).hexdigest()
    r = int(hash_value[:2], 16)
    g = int(hash_value[2:4], 16)
    b = int(hash_value[4:6], 16)
    return [r, g, b]

# Function to predict future position based on current velocity using Dead Reckoning
def dead_reckoning(kf, dt=1):
    """
    Predicts future position based on current velocity using Dead Reckoning.
    Assumes the state vector format is [x, y, vx, vy].T.

    Parameters:
        kf (KalmanFilter): Kalman filter object containing the state vector.
        dt (float, optional): Time step for predicting future position. Default is 1.

    Returns:
        tuple: Future position coordinates (future_x, future_y).
    """
    x, y, vx, vy = kf.x.flatten()
    future_x = x + (vx * dt)
    future_y = y + (vy * dt)
    return int(future_x), int(future_y)

# Function to play a beep sound as an alert
def beep_alert(frequency=2500, duration=1000):
    """
    Plays a beep sound as an alert.

    Parameters:
        frequency (int, optional): Frequency of the beep sound in Hertz. Default is 2500.
        duration (int, optional): Duration of the beep sound in milliseconds. Default is 1000 (1 second).
    """
    winsound.Beep(frequency, duration)

# Function to check if any two detections are within a specified proximity threshold
def check_proximity(detections, threshold=50):
    """
    Checks if any two detections are within a specified proximity threshold.

    Args:
        detections (list): List of detections, where each detection is represented as [x1, y1, x2, y2, ...].
        threshold (float): Maximum distance threshold for considering two detections as close.

    Returns:
        bool: True if any two detections are close to each other, False otherwise.
    """
    num_detections = len(detections)
    for i in range(num_detections):
        for j in range(i + 1, num_detections):
            x1, y1, _, _, _, _, _ = detections[i]
            x2, y2, _, _, _, _, _ = detections[j]
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if distance < threshold:
                return True
    return False


def consolidate_detections(detections, iou_threshold=0.5):
    """
    Consolidates detections by merging overlapping bounding boxes based on IoU threshold.

    Parameters:
        detections (list): List of detections, where each detection is [x1, y1, x2, y2, confidence, class_id, class_name].
        iou_threshold (float): Threshold for IoU to consider two detections as overlapping.

    Returns:
        list: Consolidated list of detections.
    """
    if not detections:
        return []

    # Initialize list to keep track of merged detections
    consolidated = []

    # Sort detections by confidence score in descending order
    detections.sort(key=lambda x: x[4], reverse=True)

    # Array to keep track of whether a detection has been used
    used = [False] * len(detections)

    for i in range(len(detections)):
        if not used[i]:
            # Mark this detection as used
            used[i] = True
            x1, y1, x2, y2, conf, cls_id, class_name = detections[i]
            # Initialize merged area as the area of the current detection
            cx1, cy1, cx2, cy2 = x1, y1, x2, y2

            # Look for overlapping detections to merge
            for j in range(i + 1, len(detections)):
                if not used[j]:
                    nx1, ny1, nx2, ny2, nconf, ncls_id, nclass_name = detections[j]

                    # Calculate the IoU
                    inter_x1 = max(cx1, nx1)
                    inter_y1 = max(cy1, ny1)
                    inter_x2 = min(cx2, nx2)
                    inter_y2 = min(cy2, ny2)
                    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    area1 = (cx2 - cx1) * (cy2 - cy1)
                    area2 = (nx2 - nx1) * (ny2 - ny1)
                    union_area = area1 + area2 - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0

                    # If IoU exceeds the threshold, merge the detections
                    if iou >= iou_threshold:
                        used[j] = True  # Mark detection as used
                        # Merge the bounding boxes
                        cx1, cy1 = min(cx1, nx1), min(cy1, ny1)
                        cx2, cy2 = max(cx2, nx2), max(cy2, ny2)
                        conf = max(conf, nconf)  # Take the higher confidence

            # Add merged detection to the consolidated list
            consolidated.append([cx1, cy1, cx2, cy2, conf, cls_id, class_name])

    return consolidated
