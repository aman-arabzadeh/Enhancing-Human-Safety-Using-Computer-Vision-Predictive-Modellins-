# utilsNeeded.py
import hashlib
import os
import uuid

import cv2
import csv
import logging

import winsound

from ultralytics import YOLO
import time


# This is helper file used for my both algorithms, functions needed in both
# Authorship Information
"""
Author: Koray Aman Arabzadeh
Thesis: Mid Sweden University.
Bachelor Thesis - Bachelor of Science in Engineering, Specialisation in Computer Engineering
Main field of study: Computer Engineering
Credits: 15 hp (ECTS)
Semester, Year: Spring, 2024
Supervisor: Emin Zerman
Examiner: Stefan Forsström
Course code: DT099G
Programme: Degree of Bachelor of Science with a major in Computer Engineering



Resources used: 
https://opencv.org/
https://stackoverflow.com/
https://github.com
https://pieriantraining.com/kalman-filter-opencv-python-example/
"""


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




# Function to generate unique color for each class ID
def get_color_by_id(class_id):
    hash_value = hashlib.sha256(str(class_id).encode()).hexdigest()
    r = int(hash_value[:2], 16)
    g = int(hash_value[2:4], 16)
    b = int(hash_value[4:6], 16)
    return [r, g, b]

# Function to play a beep sound as an alert

def trigger_proximity_alert( det):
    print(f"Proximity alert: {det[6]} detected near or inside the central area!")
    winsound.Beep(2500, 1000)  # Beep sound for alert


def is_object_within_bounds(det, center_area):
    """
    Determines if an object is within a specified bounding area.

    Parameters:
    - det: Detection details, expected to include the bounding box coordinates (x1, y1, x2, y2).
    - center_area: A tuple containing the coordinates for the top left and bottom right points of the center area (top_left, bottom_right).

    Returns:
    - bool: True if the object is within the center area, False otherwise.
    """
    (x1_obj, y1_obj, x2_obj, y2_obj, _, _, _) = det
    (top_left, bottom_right) = center_area
    x1_area, y1_area = top_left
    x2_area, y2_area = bottom_right

    return not (x2_obj < x1_area or x1_obj > x2_area or y2_obj < y1_area or y1_obj > y2_area)


def is_object_near_boundary(det, proximity_threshold, center_area):
    """
    Determines if an object is near the boundary of a specified area, within a given proximity threshold.

    Parameters:
    - det: Detection details, expected to include the bounding box coordinates (x1, y1, x2, y2).
    - proximity_threshold: The distance threshold that defines "nearness".
    - center_area: A tuple containing the coordinates for the top left and bottom right points of the center area (top_left, bottom_right).

    Returns:
    - bool: True if the object is near the boundary of the center area within the specified threshold, False otherwise.
    """
    (x1_obj, y1_obj, x2_obj, y2_obj, _, _, _) = det
    (top_left, bottom_right) = center_area
    x1_area, y1_area = top_left
    x2_area, y2_area = bottom_right

    # Check if the object is near the horizontal or vertical boundaries of the center area within the given threshold
    return ((x1_obj > x2_area and (x1_obj - x2_area) <= proximity_threshold) or
            (x2_obj < x1_area and (x1_area - x2_obj) <= proximity_threshold) or
            (y1_obj > y2_area and (y1_obj - y2_area) <= proximity_threshold) or
            (y2_obj < y1_area and (y1_area - y2_obj) <= proximity_threshold))


def draw_predictions(frame, det, current_x, current_y, future_x, future_y, color):
    x1, y1, x2, y2, _, cls, class_name = det
    cv2.circle(frame, (int(current_x), int(current_y)), 10, color, -1)
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    label = f"{class_name} ({cls})"
    cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.circle(frame, (int(future_x), int(future_y)), 10, (0, 255, 0), -1)




def setup_csv_writer(filename='tracking_and_predictions.csv'):
    try:
        file = open(filename, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'det_x', 'det_y', 'pred_x', 'pred_y', 'class_name'])
        return file, writer
    except IOError as e:
        logging.error(f"File operations failed: {str(e)}")
        return None, None


def highlight_center_area(frame, center_area, label="Robotic Arm", overlay=None):
    """
    Draws a rectangle on the frame around the specified center area, adds a label at the top center of the rectangle,
    and optionally overlays an image within the rectangle. If an overlay is provided, it is resized to fit the rectangle
    and blended into the frame with partial transparency.

    Parameters:
    - frame (np.array): The image (frame) on which to draw.
    - center_area (tuple): Tuple of (top_left, bottom_right) coordinates for the rectangle.
    - label (str): Text to display inside the rectangle.
    - overlay (np.array, optional): Image to overlay within the rectangle. It will be resized to fit the rectangle if provided.

    Returns:
    - np.array: The modified frame with the rectangle, label, and optional overlay.
    """
    top_left, bottom_right = center_area
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Draw the rectangle

    # Calculate label position to be at the top-center of the rectangle
    label_x = (top_left[0] + bottom_right[0]) // 2
    label_y = max(top_left[1] - 10, 10)  # Ensure label is within the frame's boundary

    # Place the label on the frame
    cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # If an overlay image is provided, adjust its size and blend it into the rectangle
    if overlay is not None:
        overlay_height = bottom_right[1] - top_left[1]
        overlay_width = bottom_right[0] - top_left[0]
        resized_overlay = cv2.resize(overlay, (overlay_width, overlay_height))
        # Blend the resized overlay into the rectangle with 50% transparency
        region_of_interest = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = cv2.addWeighted(
            region_of_interest, 0.5, resized_overlay, 0.5, 0)

    return frame




def update_center_area( frame_width, frame_height, factor=4):
    center_x, center_y = frame_width // 2, frame_height // 2
    area_width, area_height = frame_width // factor, frame_height // factor
    top_left = (center_x - area_width // 2, center_y - area_height // 2)
    bottom_right = (center_x + area_width // 2, center_y + area_height // 2)
    return (top_left, bottom_right)



def cleanup(cap, file):

    if cap:
        cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows
    if file:
        file.close()  # Close the CSV file if it's open


def load_model(model_path):
    try:
        model = YOLO(model_path)  # Attempt to load the model
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        return None



def initialize_video_capture(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error("Failed to open video source.")
        return None
    return cap

def track_object(det):
    """
    Track object across frames.
    Returns a UUID for now, but ideally would check against existing tracked objects.
    """
    return str(uuid.uuid4())

def log_detection(writer, timestamp, center_x, center_y, future_x, future_y, object_class):
    """
    Log detection details into a CSV writer.
    """
    writer.writerow([timestamp, center_x, center_y, future_x, future_y, object_class])

def is_object_near(det, center_area, proximity_threshold):
    """
    Check if an object is near based on given thresholds and area definitions.
    """
    return is_object_within_bounds(det, center_area) or is_object_near_boundary(det, proximity_threshold, center_area)

def handle_alert(alert_file,save_alert_times, det, timestamp, center_x, center_y, future_x, future_y, start_time, center_area):
    """
    Handle alert conditions and save the alert times accordingly.
    """
    hazard_time = timestamp - start_time
    alert_condition = "center" if is_object_within_bounds(det, center_area) else "Nearness"
    save_alert_times(alert_file,timestamp, det[6], center_x, center_y, future_x, future_y, hazard_time, alert_condition)

def save_alert_times(alert_file, timestamp, object_class, location_x, location_y, future_pos_x, future_pos_y, hazard_time,
                     alert_condition):
    try:
        needs_header = not os.path.exists(alert_file) or os.stat(alert_file).st_size == 0
        with open(alert_file, 'a', newline='') as file:
            writer = csv.writer(file)
            if needs_header:
                writer.writerow([
                    'Event DateTime UTC',
                    'Detected Object Type',
                    'Object Location X (px)',
                    'Object Location Y (px)',
                    'Predicted Future Location X (px)',
                    'Predicted Future Location Y (px)',
                    'Hazard Time Since Start (seconds)',
                    'Alert Type'
                ])
            writer.writerow(
                [timestamp, object_class, location_x, location_y, future_pos_x, future_pos_y, hazard_time,
                 alert_condition])
    except IOError as e:
        logging.error(f"Failed to save alert time: {str(e)}")
