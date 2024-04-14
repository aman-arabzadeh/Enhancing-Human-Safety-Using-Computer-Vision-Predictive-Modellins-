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
In this code I use openCV kalman  filter implementation and my own dead reckoning function
"""

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KalmanFilterWrapper:
    def __init__(self):
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
            elapsed_time = time.time() - self.start_time  # Calculate elapsed time since start
            center_x, center_y, kf_wrapper = self.apply_kalman_filter(det)
            self.writer.writerow([elapsed_time, center_x, center_y, kf_wrapper.future_x, kf_wrapper.future_y, det[6]])
            self.draw_predictions(frame, det, kf_wrapper)

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
