# utilsNeeded.py
import hashlib
import os

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


def check_proximity_simple(target, specific_object_detections, proximity_threshold):
    """
    Checks if any specific object detection is within a proximity threshold of the bounding box of the target.
    also checks  if any specific object detection overlaps with the bounding box of the target just in case.

    Args:
        target (list): List of detections for the target, where each detection is represented as [x1, y1, x2, y2, ...].
        specific_object_detections (list): List of detections for the specific object, where each detection is represented as [x1, y1, x2, y2, ...].
        proximity_threshold (float): Distance threshold to check for proximity.

    Returns:
        bool: True if any specific object detection is near the target within the proximity threshold, False otherwise.
    """
    for values in target:
        x1_target, y1_target, x2_target, y2_target, *_ = values
        center_target_x = (x1_target + x2_target) / 2
        center_target_y = (y1_target + y2_target) / 2

        for obj_det in specific_object_detections:
            x1_obj, y1_obj, x2_obj, y2_obj, *_ = obj_det
            center_obj_x = (x1_obj + x2_obj) / 2
            center_obj_y = (y1_obj + y2_obj) / 2

            # Calculate the Euclidean distance between centers
            distance = ((center_obj_x - center_target_x) ** 2 + (center_obj_y - center_target_y) ** 2) ** 0.5
            if distance <= proximity_threshold:
                return True
            """
            # Check if there's any intersection between the bounding boxes
            elif (x1_obj < x2_target and x2_obj > x1_target and
                    y1_obj < y2_target and y2_obj > y1_target):
                return True
            """
    return False


def check_proximity(target, specific_object_detections):
    """
    Checks if any specific object detection overlaps with the bounding box of the target.

    Args:
        person_detections (list): List of detections for the target, where each detection is represented as [x1, y1, x2, y2, ...].
        specific_object_detections (list): List of detections for the specific object, where each detection is represented as [x1, y1, x2, y2, ...].

    Returns:
        bool: True if any specific object detection overlaps with the bounding box of the target, False otherwise.
        :param specific_object_detections:
        :param target:
    """
    for values in target:
        x1_target, y1_target, x2_target, y2_target, _, _, _ = values
        for obj_det in specific_object_detections:
            x1_obj, y1_obj, x2_obj, y2_obj, _, _, _ = obj_det
            # Check if there's any intersection between the bounding boxes
            if (x1_obj < x2_target and x2_obj > x1_target and
                    y1_obj < y2_target and y2_obj > y1_target):
                return True
    return False
def check_nearness(target, specific_object_detections, proximity_threshold=10):
    """
    Checks if any specific object detection is near the bounding box of the target without overlapping.

    Args:
        target (list): List of detections for the target, where each detection is represented as [x1, y1, x2, y2].
        specific_object_detections (list): List of detections for specific objects, where each detection is represented as [x1, y1, x2, y2].
        proximity_threshold (int): The pixel distance within which objects are considered near each other.

    Returns:
        bool: True if any specific object detection is near the target within the proximity threshold, False otherwise.
    """
    for t_values in target:
        x1_target, y1_target, x2_target, y2_target, *_ = t_values

        for obj_det in specific_object_detections:
            x1_obj, y1_obj, x2_obj, y2_obj, *_ = obj_det

            # Check if bounding boxes are near without overlapping
            if (x2_obj < x1_target and (x1_target - x2_obj) <= proximity_threshold) or \
                    (x1_obj > x2_target and (x1_obj - x2_target) <= proximity_threshold) or \
                    (y2_obj < y1_target and (y1_target - y2_obj) <= proximity_threshold) or \
                    (y1_obj > y2_target and (y1_obj - y2_target) <= proximity_threshold):
                return True

    return False



# utilsNeeded.py
def draw_predictions(frame, det, current_x, current_y, future_x, future_y):
    """
    Draw bounding boxes, labels, and future position on the frame.

    Args:
        frame: Image on which to draw.
        det: Detection details containing coordinates and class info [x1, y1, x2, y2, _, cls, class_name].
        current_x, current_y: Current central coordinates of the object.
        future_x, future_y: Predicted future coordinates of the object.

    Returns:
        None: Modifies the frame directly.
    """
    x1, y1, x2, y2, _, cls, class_name = det
    color = get_color_by_id(cls)  # Existing function to get color based on class ID

    # Draw current position circle
    cv2.circle(frame, (int(current_x), int(current_y)), 10, color, -1)
    cv2.line(frame, (int(current_x), int(current_y + 20)), (int(current_x + 50), int(current_y + 20)), color, 2, 8)

    # Draw rectangle around detected object
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # Put label near the bounding box
    label = f"{class_name} ({cls})"
    cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw future position circle in green
    future_color = (0, 255, 0)  # RGB for green
    cv2.circle(frame, (int(future_x), int(future_y)), 10, future_color, -1)
    cv2.line(frame, (int(future_x), int(future_y + 20)), (int(future_x + 50), int(future_y + 20)), future_color, 2, 8)


def cleanup(cap, file):
    """
    Releases the video capture object and destroys all OpenCV windows. Closes the file if it is open.

    Args:
        cap (cv2.VideoCapture): The video capture object used to acquire frames.
        file (file object): The file object to write CSV data.

    Returns:
        None
    """
    if cap:
        cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows
    if file:
        file.close()  # Close the CSV file if it's open


def load_model(model_path):
    """
    Loads a YOLO model specified by the given path.

    Args:
        model_path (str): Path to the model file.

    Returns:
        model (YOLO): Loaded YOLO model if successful, None otherwise.

    Raises:
        logs an error if the model loading fails.
    """
    try:
        model = YOLO(model_path)  # Attempt to load the model
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        return None



def initialize_video_capture(source):
    """
    Initializes the video capture object for the given source.

    Args:
        source (int | str): The device index or the video file path.

    Returns:
        cap (cv2.VideoCapture | None): The initialized video capture object if successful, None otherwise.

    Raises:
        logs an error if video source can't be opened.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error("Failed to open video source.")
        return None
    return cap


def setup_csv_writer(filename='tracking_and_predictions.csv'):
    """
    Sets up a CSV writer for logging tracking and predictions data.

    Args:
        filename (str): The name of the file where data will be written.

    Returns:
        tuple (file, writer): A tuple containing the file object and the CSV writer.

    Raises:
        logs an error if the file operations fail.
    """
    try:
        file = open(filename, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'det_x', 'det_y', 'pred_x', 'pred_y', 'class_name'])
        return file, writer
    except IOError as e:
        logging.error(f"File operations failed: {str(e)}")
        return None, None


def check_and_alert(detections, target, file_name, elapsed_time, alert_start_time, start_time, alert_times, proximity_threshold, save_alert_times_func, check_proximity_func, check_nearness_func, beep_alert_func):
    """
    Checks each detection against a target to determine if an alert should be issued based on proximity and overlap criteria.

    Args:
        detections (list): List of detected objects.
        target (str): The class name of the target to check.
        file_name (str): The file name where alert times will be saved.
        elapsed_time (float): The current elapsed time from the start of detection.
        alert_start_time (float): The start time of the current alert, if any.
        start_time (float): The start time of the program for overall timing.
        alert_times (list): List of times when alerts have been issued.
        proximity_threshold (int): The proximity threshold for issuing alerts.
        save_alert_times_func (function): The function to call to save alert times.
        check_proximity_func (function): The function to check for object proximity.
        check_nearness_func (function): The function to check for object nearness.
        beep_alert_func (function): The function to execute an audio alert.

    Returns:
        tuple: Updated alert_start_time and alert_times.

    Side effects:
        This function can modify alert_times and issue audio alerts based on detection conditions.
    """
    person_detections = [d for d in detections if d[6] == target]
    other_objects = [d for d in detections if d[6] != target]

    alert_issued = False
    for person in person_detections:
        for obj in other_objects:
            if check_proximity_func([person], [obj]) or check_nearness_func([person], [obj], proximity_threshold):
                if alert_start_time is None:
                    alert_start_time = elapsed_time  # Log the relative time when hazard detected
                beep_alert_func(frequency=3000, duration=500)
                alert_issued_at = elapsed_time + (time.time() - start_time) - alert_start_time
                save_alert_times_func(file_name, person[6], obj[6], alert_start_time, alert_issued_at)
                alert_issued = True

    if not alert_issued and alert_start_time is not None:
        alert_duration = elapsed_time - alert_start_time
        alert_times.append((alert_start_time, alert_duration))
        alert_start_time = None  # Reset the alert start time

    return alert_start_time, alert_times


def save_alert_times(file_path, person_class, object_class, hazard_time, alert_time):
    """
    Saves the alert times into a CSV file specified by the file path.

    Args:
        file_path (str): Path to the file where alert times will be recorded.
        person_class (str): Class name of the person involved in the alert.
        object_class (str): Class name of the object involved in the alert.
        hazard_time (float): The time when the hazard was first detected.
        alert_time (float): The time when the alert was issued.

    Raises:
        logs an error if unable to write to the file.
    """
    try:
        needs_header = not os.path.exists(file_path) or os.stat(file_path).st_size == 0
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            if needs_header:
                writer.writerow(['Hazard Time', 'Alert Time', 'Person Class', 'Object Class', 'Response Time'])
            writer.writerow([hazard_time, alert_time, person_class, object_class])
    except IOError as e:
        logging.error(f"Failed to save alert time: {str(e)}")
