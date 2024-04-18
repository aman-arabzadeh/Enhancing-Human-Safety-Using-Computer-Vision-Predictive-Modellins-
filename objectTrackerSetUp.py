import os

import cap as cap
import cv2
import numpy as np
import csv
import logging
from ultralytics import YOLO
import utilsNeeded
import time  # Import time to work with timestamps
from kalmanSetUp import KalmanFilterWrapper

# Main Functionality
"""



This Python script integrates YOLOv8 for object detection, Kalman filtering for object tracking,
dead reckoning for predicting future positions, and audio alerts for detectiopn inside the person boxe zone.
This code can monitor movements around robotic arms to alert a trigger of sound to alert abbout hazards.
In this code I use openCV kalman  filter implementation and my own dead reckoning function.



"""
class ObjectTracker:
    def __init__(self, model_path, source=0):
        self.writer = None
        self.model = self.load_model(model_path)
        self.cap = self.initialize_video_capture(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.kalman_filters = {}
        self.setup_csv_writer()
        self.start_time = time.time()  # Start time of the tracking
        # Initialize other attributes
        self.alert_times = []
        self.alert_start_time = None
        self.last_positions = {}  # To store last known positions for velocity calculation

    def apply_dead_reckoning(self, det, time_step=1.0):
        """
        Apply dead reckoning to predict future position based on current velocity.
        `det`: Detection information, including current position.
        `time_step`: Future time step in seconds for prediction.
        """

        x1, y1, x2, y2, _, cls, class_name = det
        current_center_x = int((x1 + x2) / 2)
        current_center_y = int((y1 + y2) / 2)

        # Get last position and calculate velocity
        if cls in self.last_positions:
            last_x, last_y, last_time = self.last_positions[cls]
            print(f'last_time is {last_time}')
            time_delta = time.time() - last_time

            # Ensuring so that time_delta is not zero to avoid division by zero,
            if time_delta > 0.001:  # Using 0.001 seconds as a minimal delta
                print(f'time delta is {time_delta}')
                velocity_x = (current_center_x - last_x) / time_delta
                velocity_y = (current_center_y - last_y) / time_delta
            else:
                velocity_x, velocity_y = 0, 0  # Default to zero velocity if time_delta is too small
        else:
            velocity_x, velocity_y = 0, 0  # Assume starting velocity is zero

        # Predict future position using dead reckoning
        future_x = int(current_center_x + velocity_x * time_step)
        future_y = int(current_center_y + velocity_y * time_step)

        # Update last positions
        self.last_positions[cls] = (current_center_x, current_center_y, time.time())

        return current_center_x, current_center_y, future_x, future_y

    def track_objects_DR(self, frame, detections):
        for det in detections:
            elapsed_time = time.time() - self.start_time
            center_x, center_y, future_x, future_y = self.apply_dead_reckoning(det)
            # Write current and predicted positions
            self.writer.writerow([elapsed_time, center_x, center_y, future_x, future_y, det[6]])
            self.draw_predictions(frame, det, center_x, center_y)
           # self.check_and_alert(detections, elapsed_time)
    def load_model(self, model_path):
        try:
            model = YOLO(model_path)
            logging.info("Model loaded successfully.")
            return model
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            exit(1)

    def initialize_video_capture(self, source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logging.error("Failed to open video source.")
            exit(1)
        return cap

    def setup_csv_writer(self):
        try:
            self.file = open('tracking_and_predictions.csv', 'w', newline='')
            self.writer = csv.writer(self.file)
            self.writer.writerow(['timestamp', 'det_x', 'det_y', 'pred_x', 'pred_y', 'class_name'])
        except IOError as e:
            logging.error(f"File operations failed: {str(e)}")
            exit(1)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            consolidated_detections = utilsNeeded.run_yolov8_inference(self.model, frame)
            # consolidated_detections = utilsNeeded.consolidate_detections(detections)
            #self.track_objects(frame, consolidated_detections)
            self.track_objects_DR(frame, consolidated_detections)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cleanup()

    def track_objects(self, frame, detections):
        for det in detections:
            elapsed_time = time.time() - self.start_time
            center_x, center_y, kf_wrapper, _, _ = self.apply_kalman_filter(det)
            # Predict future position
            future_x, future_y = kf_wrapper.predict2()  # Get future predictions
            # Write current and predicted positions
            self.writer.writerow([elapsed_time, center_x, center_y, future_x, future_y, det[6]])
            self.draw_predictions(frame, det, center_x, center_y)
            #self.check_and_alert(detections, elapsed_time)

    def check_and_alert(self, detections, elapsed_time):
        person_detections = [d for d in detections if d[6] == 'person']
        other_objects = [d for d in detections if d[6] != 'person']

        for person in person_detections:
            for obj in other_objects:
                if utilsNeeded.check_proximity([person], [obj]) or utilsNeeded.check_nearness([person], [obj]):  # Check proximity or near check bounding box
                    if self.alert_start_time is None:
                        self.alert_start_time = elapsed_time  # Log the relative time when hazard detected
                    utilsNeeded.beep_alert(frequency=3000, duration=500)
                    alert_issued_at = elapsed_time + (time.time() - self.start_time) - self.alert_start_time
                    self.save_alert_times(person[6], obj[6], self.alert_start_time, alert_issued_at)

        if not utilsNeeded.check_proximity(person_detections, other_objects) and self.alert_start_time is not None:
            # Calculate the duration of the alert if needed, and reset alert_start_time
            alert_duration = elapsed_time - self.alert_start_time
            self.alert_times.append((self.alert_start_time, alert_duration))
            self.alert_start_time = None  # Reset the alert start time

    """

    Hazard Time:
    This is the timestamp at which a potential hazard is first detected by the system. 


    Alert Time:
    This is the timestamp when an alert is actually issued by the system in response to the detected hazard. 

    Person Class:
    This refers to the classification of the detected entity as a "person" in the object detection system.

    Object Class:
    This describes the type of object that has been detected as posing a potential hazard when in close proximity to a person. 

    Response Time:
    This is calculated as the difference between the "Alert Time" and the "Hazard Time". 
    It measures how quickly the system responds to a detected hazard by issuing an alert. 
    A shorter response time indicates a more effective and potentially safer system,
     as it reduces the window of risk for accidents or injuries.


    """

    def save_alert_times(self, person_class, object_class, hazard_time, alert_time):
        file_path = 'alert_times.csv'  # Define the file path
        try:
            # Check if the file needs a header (new or empty file)
            needs_header = not os.path.exists(file_path) or os.stat(file_path).st_size == 0

            with open(file_path, 'a', newline='') as file:
                writer = csv.writer(file)

                # Write the data row (all times are in seconds)

                if needs_header:
                    writer.writerow(['Hazard Time', 'Alert Time', 'Person Class', 'Object Class', 'Response Time'])

                # Write the data row (all times are in seconds)

                writer.writerow([hazard_time, alert_time, person_class, object_class, alert_time - hazard_time])

        except IOError as e:
            logging.error(f"Failed to save alert time: {str(e)}")

    def apply_kalman_filter(self, det):
        x1, y1, x2, y2, _, cls, class_name = det
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        # Check if a Kalman filter already exists for this class; if not, create and initialize one
        if cls not in self.kalman_filters:
            self.kalman_filters[cls] = KalmanFilterWrapper()
            # Initialize with the first detected center position and assume initial velocities are zero
            self.kalman_filters[cls].initialize(center_x, center_y)
        kf_wrapper = self.kalman_filters[cls]
        measurement = np.array([[center_x], [center_y]], np.float32)
        kf_wrapper.predict2()
        kf_wrapper.correct(measurement)
        update_x = center_x
        update_y = center_y
        return center_x, center_y, kf_wrapper, update_x, update_y

    def draw_predictions(self, frame, det, x,y):
        x1, y1, x2, y2, _, cls, class_name = det
        color = utilsNeeded.get_color_by_id(int(cls))
        cv2.circle(frame, (int(x), int(y)), 10, color, -1)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{class_name} ({cls})"
        cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.file.close()
        logging.info("Cleaned up resources and exited.")

