import cv2 as cv
import cv2
import numpy as np
import csv
import logging
import utilsNeeded
import time  # Import time to work with timestamps
from kalmanSetUp import KalmanFilterWrapper
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

class ObjectTracker_Kalman:
    """
    This class implements an object tracking system using YOLOv8 for object detection,
    Kalman filtering for object tracking, and proximity-based audio alerts. It is designed
    to monitor movements around robotic arms in industrial settings, alerting to potential hazards.

    Attributes:
        writer (csv.writer): CSV writer object for logging predictions.
        target (str): The class of the target object to track and monitor.
        filename_prediction (str): Path to save prediction tracking data as CSV.
        file_name_alert (str): Path to save alert times data as CSV.
        proximity_threshold (int): The distance threshold to consider for proximity alerts.
        model (YOLO): The YOLOv8 model loaded for object detection.
        cap (cv2.VideoCapture): Video capture object for frame acquisition.
        fps (float): Frames per second of the video source.
        kalman_filters (dict): Dictionary storing KalmanFilterWrapper instances by class ID.
        file (file object): File object for the CSV writer.
        start_time (float): Start time of the tracking to calculate elapsed time.
        alert_times (list): List of times when alerts were issued.
        alert_start_time (float|None): Start time of the current alert period.
        last_positions (dict): Dictionary storing last known positions of detected objects.

    Methods:
        run(): Main method to start the tracking and detection loop.
        process_detections(detections, frame): Processes each detection per frame.
        apply_kalman_filter(det): Applies Kalman filtering to smooth and predict object positions.
    """
    def __init__(self, model_path, proximity_threshold, file_name_predict, file_name_alert, target, source=0):
        """
        Initializes the object tracker with necessary parameters and setups.

        Parameters:
            model_path (str): Path to the YOLOv8 model for object detection.
            proximity_threshold (int): Pixel threshold to determine when objects are considered close to each other.
            file_name_predict (str): Filename to save prediction data.
            file_name_alert (str): Filename to save alert data.
            target (str): Target object class name to monitor specifically.
            source (int|str): Video source, default is the first camera.
        """
        self.writer = None
        self.target = target
        self.filename_prediction = file_name_predict
        self.file_name_alert = file_name_alert
        self.proximity_threshold = proximity_threshold
        self.model = utilsNeeded.load_model(model_path)
        self.cap = utilsNeeded.initialize_video_capture(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.kalman_filters = {}
        self.file, self.writer = utilsNeeded.setup_csv_writer(self.filename_prediction)
        self.start_time = time.time()
        self.alert_times = []
        self.alert_start_time = None
        self.last_positions = {}

    def run(self):
        """
        Runs the main loop to capture frames and process detections. Displays the frame and checks for exit command.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to capture frame. Exiting...")
                break
            detections = utilsNeeded.run_yolov8_inference(self.model, frame)
            self.process_detections(detections, frame)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        utilsNeeded.cleanup(self.cap, self.file)

    def process_detections(self, detections, frame):
        """
        Processes each detection from YOLOv8, applies Kalman filtering, predicts future positions, and logs data.

        Parameters:
            detections (list): List of detections from the YOLOv8 model.
            frame (np.array): Current frame from the video source.
        """
        for det in detections:
            elapsed_time = time.time() - self.start_time
            center_x, center_y, kf_wrapper = self.apply_kalman_filter(det)
            future_x, future_y = kf_wrapper.predict()  # Get future position before correction
            kf_wrapper.correct(np.array([[center_x], [center_y]], np.float32))
            self.writer.writerow([elapsed_time, center_x, center_y, future_x, future_y, det[6]])
            utilsNeeded.draw_predictions(frame, det, center_x, center_y, future_x, future_y)

    def apply_kalman_filter(self, det):
        """
        Applies a Kalman Filter to the given detection to estimate and predict the object's position.

        Parameters:
            det (list): Detection data including bounding box and class info.

        Returns:
            tuple: Current estimated center position (x, y) and the Kalman filter wrapper instance.
        """
        x1, y1, x2, y2, _, cls, class_name = det
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        if cls not in self.kalman_filters:
            self.kalman_filters[cls] = KalmanFilterWrapper()
            self.kalman_filters[cls].initialize(center_x, center_y)
        return center_x, center_y, self.kalman_filters[cls]

    
    
if __name__ == "__main__":
    tracker = ObjectTracker_Kalman(
        'yolov8n.pt',
        source=0,
        proximity_threshold=20,
        file_name_predict='tracking_and_predictions.csv',
        file_name_alert='alert_times.csv',
        target='person'
    )
    tracker.run()