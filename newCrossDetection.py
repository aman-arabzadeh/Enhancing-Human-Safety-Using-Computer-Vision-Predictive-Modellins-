import os

import cap as cap
import cv2
import numpy as np
import csv
import logging
from ultralytics import YOLO
import utilsNeeded
import time  # Import time to work with timestamps

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

# Main Functionality
"""



This Python script integrates YOLOv8 for object detection, Kalman filtering for object tracking,
dead reckoning for predicting future positions, and audio alerts for detectiopn inside the person boxe zone.
This code can monitor movements around robotic arms to alert a trigger of sound to alert abbout hazards.
In this code I use openCV kalman  filter implementation and my own dead reckoning function.



"""

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class KalmanFilterWrapper:
    def __init__(self):
        self.future_x = None
        self.future_y = None
        self.kf = cv2.KalmanFilter(4, 2)  # State: [x, y, dx, dy], Measurement: [x, y]
        self.kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.base_process_noise = np.eye(4, dtype=np.float32) * 0.03
        self.kf.processNoiseCov = self.base_process_noise

    def correct(self, measurement):
        self.kf.correct(measurement)

    def predict(self, fps, velocity_scale=1.0):
        self.kf.processNoiseCov = self.base_process_noise * velocity_scale
        prediction = self.kf.predict()
        current_predicted_x = prediction[0, 0]
        current_predicted_y = prediction[1, 0]
        velocity_x = prediction[2, 0]
        velocity_y = prediction[3, 0]
        dt = 1 / fps
        self.future_x = current_predicted_x + velocity_x * dt
        self.future_y = current_predicted_y + velocity_y * dt
        return prediction


class ObjectTracker:
    def __init__(self, model_path, source=0):
        self.model = self.load_model(model_path)
        self.cap = self.initialize_video_capture(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.kalman_filters = {}
        self.setup_csv_writer()
        self.start_time = time.time()  # Start time of the tracking
        # Initialize other attributes
        self.alert_times = []
        self.alert_start_time = None


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
            detections = utilsNeeded.run_yolov8_inference(self.model, frame)
            consolidated_detections = utilsNeeded.consolidate_detections(detections)
            self.track_objects(frame, consolidated_detections)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cleanup()

    def track_objects(self, frame, detections):
        for det in detections:
            elapsed_time = time.time() - self.start_time
            center_x, center_y, kf_wrapper = self.apply_kalman_filter(det)
            self.writer.writerow([elapsed_time, center_x, center_y, kf_wrapper.future_x, kf_wrapper.future_y, det[6]])
            self.draw_predictions(frame, det, kf_wrapper)
            self.check_and_alert(detections, elapsed_time)  # Pass elapsed time to check_and_alert

    def check_and_alert(self, detections, elapsed_time):
        person_detections = [d for d in detections if d[6] == 'person']
        other_objects = [d for d in detections if d[6] != 'person']

        for person in person_detections:
            for obj in other_objects:
                if utilsNeeded.check_proximity([person], [obj]):  # Check proximity
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
        if cls not in self.kalman_filters:
            self.kalman_filters[cls] = KalmanFilterWrapper()
        kf_wrapper = self.kalman_filters[cls]
        measurement = np.array([[center_x], [center_y]], np.float32)
        kf_wrapper.correct(measurement)
        kf_wrapper.predict(self.fps)
        return center_x, center_y, kf_wrapper

    def draw_predictions(self, frame, det, kf_wrapper):
        x1, y1, x2, y2, _, cls, class_name = det
        color = utilsNeeded.get_color_by_id(int(cls))
        cv2.circle(frame, (int(kf_wrapper.future_x), int(kf_wrapper.future_y)), 10, color, -1)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{class_name} ({cls})"
        cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.file.close()
        logging.info("Cleaned up resources and exited.")


if __name__ == "__main__":
    tracker = ObjectTracker('yolov8n.pt')
    tracker.run()
