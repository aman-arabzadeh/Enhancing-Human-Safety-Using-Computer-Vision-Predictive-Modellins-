"""
utilitiesHelper.py

This module contains helper functions and utilities used across various algorithms for object tracking and processing.
It includes functions for image processing, sound alerts, and data logging.

Dependencies:
- OpenCV (cv2)
- hashlib, os, uuid: Standard Python libraries for hashing, operating system interaction, and unique identifier generation.
- csv: For CSV file operations.
- logging: For logging status and error messages.
- winsound: For playing sound on Windows.
- ultralytics.YOLO: For object detection tasks.
- time: For timing and performance measurement.

Authorship Information:
- Author: Koray Aman Arabzadeh
- Affiliation: Mid Sweden University
- Thesis: Bachelor Thesis in the Degree of Bachelor of Science with a major in Computer Engineering
- Title: "Implementing and Evaluating Tracking Algorithms"
- Main field of study: Computer Engineering
- Credits: 15 hp (ECTS)
- Semester, Year: Spring, 2024
- Supervisor: Emin Zerman
- Examiner: Stefan Forsström
- Course code: DT099G

Resources:
- OpenCV: https://opencv.org/
- Stack Overflow: https://stackoverflow.com/
- GitHub: https://github.com
- Pierian Training's Kalman Filter example: https://pieriantraining.com/kalman-filter-opencv-python-example/

Purpose:
This file is intended to centralize common utility functions that are required by multiple components of the tracking system.
It aims to reduce code duplication and foster reusability of common tasks such as image manipulation, data logging, and alert management.
"""

# Import necessary libraries
import hashlib
import os
import cv2
import csv
import logging
import winsound
from ultralytics import YOLO

# Place the function definitions below...



def get_color_by_id(class_id):
    """
    Generates a color based on the hash of the class ID.

    Parameters:
    - class_id (int): Class ID used to generate a unique color.

    Returns:
    - list: A list of RGB color values.
    """
    hash_value = hashlib.sha256(str(class_id).encode()).hexdigest()
    r = int(hash_value[:2], 16)  # Extract two characters for red
    g = int(hash_value[2:4], 16)  # Extract two characters for green
    b = int(hash_value[4:6], 16)  # Extract two characters for blue
    return [r, g, b]

def trigger_proximity_alert(duration=2000, freq=1000):
    """
    Triggers a beep sound to alert proximity.

    Parameters:
    - duration (int): Duration of the beep sound in milliseconds.
    - freq (int): Frequency of the beep sound in hertz.
    """
    winsound.Beep(freq, duration)  # Play a beep sound


def is_object_within_bounds(det, center_area):
    """
    Checks if an object is within the specified area.

    Parameters:
    - det (tuple): Bounding box of the object (x1, y1, x2, y2, _, _, _).
    - center_area (tuple): A tuple containing the top-left and bottom-right coordinates of the area.

    Returns:
    - bool: True if the object is within bounds, False otherwise.
    """
    (x1_obj, y1_obj, x2_obj, y2_obj, _, _, _) = det
    (top_left, bottom_right) = center_area
    x1_area, y1_area = top_left
    x2_area, y2_area = bottom_right
    # Check if the object is completely outside the area
    return not (x2_obj < x1_area or x1_obj > x2_area or y2_obj < y1_area or y1_obj > y2_area)

def is_object_near_boundary(det, proximity_threshold, center_area):
    """
    Checks if an object is near the boundary of the specified area within a given threshold.

    Parameters:
    - det (tuple): Bounding box of the object (x1, y1, x2, y2, _, _, _).
    - proximity_threshold (int): Distance threshold to check for nearness to the boundary.
    - center_area (tuple): A tuple containing the top-left and bottom-right coordinates of the area.

    Returns:
    - bool: True if the object is near the boundary, False otherwise.
    """
    (x1_obj, y1_obj, x2_obj, y2_obj, _, _, _) = det
    (top_left, bottom_right) = center_area
    x1_area, y1_area = top_left
    x2_area, y2_area = bottom_right
    # Check if the object is within the proximity threshold from the boundary
    return ((x1_obj > x2_area and (x1_obj - x2_area) <= proximity_threshold) or
            (x2_obj < x1_area and (x1_area - x2_obj) <= proximity_threshold) or
            (y1_obj > y2_area and (y1_obj - y2_area) <= proximity_threshold) or
            (y2_obj < y1_area and (y1_area - y2_obj) <= proximity_threshold))

def draw_predictions(frame, det, current_x, current_y, future_x, future_y, color):
    """
    Draws the current and predicted positions of detected objects on the frame.

    Parameters:
    - frame (np.array): The current video frame.
    - det (tuple): Detection data including bounding box coordinates.
    - current_x, current_y (int): Current center coordinates of the object.
    - future_x, future_y (int): Predicted future center coordinates of the object.
    - color (tuple): Color to use for drawing.

    Returns:
    - np.array: The frame with drawings.
    """
    x1, y1, x2, y2, _, cls, class_name = det
    cv2.circle(frame, (int(current_x), int(current_y)), 10, color, -1)  # Draw circle at current position
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # Draw bounding box
    label = f"{class_name} ({cls})"  # Label format
    if class_name == 'sports ball':
        color = (255, 0, 0)  # Blue
    elif class_name == 'frisbee':
        color = (0, 0, 255)  # Red
    else:
        color = (255, 255, 255)  # White for unknown classes

    cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Draw label
    cv2.circle(frame, (int(future_x), int(future_y)), 10, color, -1)  # Draw circle at predicted position


def setup_csv_writer(filename='tracking_and_predictions.csv'):
    """
    Sets up a CSV writer for logging detection and prediction data.

    Parameters:
    - filename (str): Path to the CSV file.

    Returns:
    - tuple: A file object and a csv.writer object or (None, None) if an error occurs.
    """
    try:
        file = open(filename, 'w', newline='')  # Open the file for writing
        writer = csv.writer(file)  # Create a CSV writer object
        writer.writerow(['timestamp', 'det_x', 'det_y', 'pred_x', 'pred_y', 'class_name'])  # Write header row
        return file, writer
    except IOError as e:
        logging.error(f"File operations failed: {str(e)}")
        return None, None
def highlight_center_area(frame, center_area, label="Robotic Arm", overlay=None):
    """
    Draws a highlighted area on the frame, optionally with a label and an overlay image.

    Parameters:
    - frame (np.array): The image frame on which to draw.
    - center_area (tuple): A tuple containing the top-left and bottom-right coordinates of the area.
    - label (str): The label to display on the highlighted area.
    - overlay (np.array, optional): An image to overlay within the highlighted area.

    Returns:
    - np.array: The modified frame with the highlighted area.
    """
    top_left, bottom_right = center_area
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Draw a green rectangle around the center area

    # Calculate position for the label to ensure it appears above the rectangle
    label_x = (top_left[0] + bottom_right[0]) // 2
    label_y = max(top_left[1] - 10, 10)  # Ensure the label does not go outside the top boundary

    cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    if overlay is not None:
        # Resize and overlay an image on the highlighted area with 50% transparency
        overlay_height = bottom_right[1] - top_left[1]
        overlay_width = bottom_right[0] - top_left[0]
        resized_overlay = cv2.resize(overlay, (overlay_width, overlay_height))
        region_of_interest = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = cv2.addWeighted(region_of_interest, 0.5, resized_overlay, 0.5, 0)

    return frame

def update_center_area(frame_width, frame_height, factor=4):
    """
    Calculates the central area of the frame based on a scaling factor.

    Parameters:
    - frame_width (int): Width of the frame.
    - frame_height (int): Height of the frame.
    - factor (int): Divisor to reduce the dimensions of the central area.

    Returns:
    - tuple: Top-left and bottom-right coordinates of the central area.
    """
    center_x, center_y = frame_width // 2, frame_height // 2
    area_width, area_height = frame_width // factor, frame_height // factor
    top_left = (center_x - area_width // 2, center_y - area_height // 2)
    bottom_right = (center_x + area_width // 2, center_y + area_height // 2)
    return (top_left, bottom_right)

def log_detection(writer, timestamp, center_x, center_y, future_x, future_y, object_class):
    """
    Logs the detection and prediction data to a CSV file.

    Parameters:
    - writer (csv.writer): Writer object to log data.
    - timestamp (float): Current time as a timestamp.
    - center_x, center_y (int): Current center coordinates of the object.
    - future_x, future_y (int): Predicted future coordinates of the object.
    - object_class (str): Class of the detected object.
    """
    writer.writerow([timestamp, center_x, center_y, future_x, future_y, object_class])

def is_object_near(det, center_area, proximity_threshold):
    """
    Checks if a detected object is near a specified area or its boundary.

    Parameters:
    - det (list): Detection data for the object.
    - center_area (tuple): Center area of interest.
    - proximity_threshold (int): Threshold distance to determine 'nearness'.

    Returns:
    - bool: True if the object is near the center area or its boundary, False otherwise.
    """
    return is_object_within_bounds(det, center_area) or is_object_near_boundary(det, proximity_threshold, center_area)

def handle_alert(alert_file, save_alert_times, det, pre_alert_time, post_alert_time, center_x, center_y, future_x, future_y, start_time, center_area):
    """
    Handles the alert process by logging the alert details based on the object's proximity to the center area.

    Parameters:
    - alert_file (str): File path to save alert logs.
    - save_alert_times (function): Function to log the alert times to the file.
    - det (list): Detection data of the object.
    - pre_alert_time, post_alert_time (float): Timestamps before and after the alert.
    - center_x, center_y, future_x, future_y (int): Current and future coordinates of the object.
    - start_time (float): Start time of the tracking process.
    - center_area (tuple): Central area of interest.
    """
    hazard_time = post_alert_time - start_time
    alert_condition = "center" if is_object_within_bounds(det, center_area) else "Nearness"
    save_alert_times(alert_file, pre_alert_time, post_alert_time, det[6], center_x, center_y, future_x, future_y, hazard_time, alert_condition, center_area)

def save_alert_times(alert_file, pre_alert_time, post_alert_time, object_class, location_x, location_y, future_pos_x, future_pos_y, hazard_time, alert_condition, center_area):
    """
    Saves detailed alert times and conditions to a CSV file.

    Parameters:
    - alert_file (str): File path for logging alert times.
    - pre_alert_time, post_alert_time (float): Start and end times of the alert.
    - object_class (str): Class of the detected object.
    - location_x, location_y, future_pos_x, future_pos_y (int): Current and predicted locations of the object.
    - hazard_time (float): Time since the start of the tracking to the hazard occurrence.
    - alert_condition (str): Type of alert condition ('center' or 'Nearness').
    - center_area (tuple): Coordinates of the center area where alerts are monitored.
    """
    alert_duration = post_alert_time - pre_alert_time  # Calculate the duration of the alert
    try:
        # Check if file exists and needs a header
        needs_header = not os.path.exists(alert_file) or os.stat(alert_file).st_size == 0
        with open(alert_file, 'a', newline='') as file:
            writer = csv.writer(file)
            if needs_header:
                writer.writerow([
                    'Pre-alert DateTime UTC',
                    'Post-alert DateTime UTC',
                    'Alert Duration (seconds)',
                    'Detected Object Type',
                    'Object Location X1 (px)',
                    'Object Location Y1 (px)',
                    'Object Location X2 (px)',
                    'Object Location Y2 (px)',
                    'Hazard Time Since Start (seconds)',
                    'Alert Type',
                    'Center Area Top-Left X (px)',
                    'Center Area Top-Left Y (px)',
                    'Center Area Bottom-Right X (px)',
                    'Center Area Bottom-Right Y (px)'
                ])
            writer.writerow([
                pre_alert_time, post_alert_time, alert_duration, object_class, location_x, location_y, future_pos_x, future_pos_y, hazard_time, alert_condition,
                center_area[0][0], center_area[0][1], center_area[1][0], center_area[1][1]
            ])
    except IOError as e:
        logging.error(f"Failed to save alert time: {str(e)}")




##########################    DEFAULT FUNCTIONS NEEDED    ##########################

def cleanup(cap, file):
    """
    Cleans up the resources used by the application.

    Parameters:
    - cap (cv2.VideoCapture): The video capture object to be released.
    - file (file object): The file object to be closed, typically used for logging data.
    """
    if cap:
        cap.release()  # Release the video capture object to free up system resources
    cv2.destroyAllWindows()  # Close all OpenCV windows to ensure no GUI remnants are left
    if file:
        file.close()  # Ensure the file is closed properly to prevent data corruption or loss


def load_model(model_path):
    """
    Attempts to load the specified model from the given path.

    Parameters:
    - model_path (str): The path to the model file.

    Returns:
    - model (YOLO): Loaded YOLO model object if successful, None otherwise.
    """
    try:
        model = YOLO(model_path)  # Attempt to load the YOLO model specified by the path
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")  # Log any exceptions raised during the model loading
        return None


def initialize_video_capture(source):
    """
    Initializes a video capture object from the specified source.

    Parameters:
    - source (int or str): The device index or the path to a video file.

    Returns:
    - cap (cv2.VideoCapture): Initialized video capture object, or None if initialization fails.
    """
    cap = cv2.VideoCapture(source)  # Create a video capture object using the provided source
    if not cap.isOpened():  # Check if the video capture object was successfully initialized
        logging.error("Failed to open video source.")  # Log an error if the video source cannot be opened
        return None
    return cap  # Return the video capture object


def run_yolov8_inference(model, frame):
    """
    Performs object detection on the given frame using the YOLOv8 model.

    Parameters:
    - model (YOLO): The YOLO model used for performing inference.
    - frame (np.array): The video frame to be processed.

    Returns:
    - detections (list): A list of detections, each detection is a list containing bounding box coordinates,
      confidence score, class ID, and class name.
    """
    # Perform inference with the YOLOv8 model
    results = model.predict(frame)
    detections = []  # Initialize an empty list to store detections

    # Process the results of the inference
    if results:
        detection_result = results[0]  # Assuming the first item in results contains the detection information
        xyxy = detection_result.boxes.xyxy.numpy()  # Extract bounding box coordinates
        confidence = detection_result.boxes.conf.numpy()  # Extract confidence scores
        class_ids = detection_result.boxes.cls.numpy().astype(int)  # Extract class IDs
        class_names = [model.model.names[cls_id] for cls_id in class_ids]  # Map class IDs to their respective names

        # Create a list of detections from the results
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, xyxy[i])  # Convert bounding box coordinates to integers
            conf = confidence[i]  # Confidence score for this detection
            cls_id = class_ids[i]  # Class ID for this detection
            class_name = class_names[i]  # Class name for this detection
            detections.append([x1, y1, x2, y2, conf, cls_id, class_name])

    return detections  # Return the list of detections


def log_detection_data(det, file_path='yolo_data.csv'):
    """
    Logs detection data to a CSV file. It writes the headers if the file is newly created or empty.

    Parameters:
        det (list or tuple): A list or tuple where det[4] is the confidence score and det[6] is the class name.
        file_path (str): The path to the CSV file where data will be logged.
    """
    # Check if the file already exists and has content
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        # Open the file for writing and write headers
        with open(file_path, 'w') as file:
            file.write('Confidence Score,Class Name\n')  # Write the column headers

    # Now, write the detection data
    with open(file_path, 'a') as file:
        # Write the confidence score and class name from the detection data
        file.write(f"{det[4]},{det[6]}\n")