import cv2
import numpy as np
import csv
import logging
import time
from utilsNeeded import load_model, initialize_video_capture, run_yolov8_inference, draw_predictions, cleanup, \
    setup_csv_writer, check_and_alert, save_alert_times, check_proximity, check_nearness, beep_alert

class DeadReckoningTracker:
    """
    This class implements object tracking using YOLOv8 for object detection and dead reckoning for predicting future positions.
    It is designed to monitor movements around robotic arms in industrial settings, alerting to potential hazards.

    Attributes:
        writer (csv.writer): CSV writer object for logging predictions.
        target (str): The class of the target object to track and monitor.
        filename_prediction (str): Path to save prediction tracking data as CSV.
        file_name_alert (str): Path to save alert times data as CSV.
        proximity_threshold (int): The distance threshold to consider for proximity alerts.
        model (YOLO): The YOLOv8 model loaded for object detection.
        cap (cv2.VideoCapture): Video capture object for frame acquisition.
        file (file object): File object for the CSV writer.
        start_time (float): Start time of the tracking to calculate elapsed time.
        last_positions (dict): Dictionary storing last known positions of detected objects.
        alert_start_time (float|None): Start time of the current alert period.
        alert_times (list): List of times when alerts were issued.
    """
    def __init__(self, model_path, proximity_threshold, file_name_predict, file_name_alert, target, source=0):
        """
        Initializes the object tracker with necessary parameters and setups.
        """
        self.target = target
        self.filename_prediction = file_name_predict
        self.file_name_alert = file_name_alert
        self.proximity_threshold = proximity_threshold
        self.model = load_model(model_path)
        self.cap = initialize_video_capture(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.file, self.writer = setup_csv_writer(self.filename_prediction)
        self.start_time = time.time()
        self.last_positions = {}
        self.alert_start_time = None
        self.alert_times = []



    def run(self):
        """
        Captures frames from the video source, runs object detection, applies dead reckoning to predict future positions,
        and logs the data to a CSV file.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to capture frame. Exiting...")
                break

            detections = run_yolov8_inference(self.model, frame)
            self.process_detections(detections, frame)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cleanup(self.cap, self.file)



    def process_detections(self, detections, frame):
        """
        Processes each detection from YOLOv8, applies dead reckoning, predicts future positions, and logs data.
        """
        timestamp = time.time()
        for det in detections:
            class_id = det[5]
            current_x, current_y, future_x, future_y = self.apply_dead_reckoning(det, timestamp)
            self.writer.writerow([timestamp, class_id, current_x, current_y, future_x, future_y, det[6]])
            draw_predictions(frame, det, current_x, current_y, future_x, future_y)

            # After processing the detection, check and alert if necessary
            self.alert_start_time, self.alert_times = check_and_alert(
                detections=detections,
                target=self.target,
                file_name=self.file_name_alert,
                elapsed_time=timestamp - self.start_time,
                alert_start_time=self.alert_start_time,
                start_time=self.start_time,
                alert_times=self.alert_times,
                proximity_threshold=self.proximity_threshold,
                save_alert_times_func=save_alert_times,
                check_proximity_func=check_proximity,
                check_nearness_func=check_nearness,
                beep_alert_func=beep_alert
            )



    def apply_dead_reckoning(self, det, timestamp):
        """
        Applies dead reckoning to predict future positions based on the current and last known positions.
        """
        x1, y1, x2, y2, _, cls, _ = det
        current_x = int((x1 + x2) / 2)
        current_y = int((y1 + y2) / 2)
        last_info = self.last_positions.get(cls, (current_x, current_y, timestamp))

        # Calculate velocities
        time_delta = timestamp - last_info[2]
        velocity_x = (current_x - last_info[0]) / time_delta if time_delta > 0 else 0
        velocity_y = (current_y - last_info[1]) / time_delta if time_delta > 0 else 0

        # Predict future position
        future_x = int(current_x + velocity_x * time_delta)
        future_y = int(current_y + velocity_y * time_delta)

        # Update last known positions
        self.last_positions[cls] = (current_x, current_y, timestamp)

        return current_x, current_y, future_x, future_y


if __name__ == "__main__":
    tracker = DeadReckoningTracker(
        model_path='yolov8n.pt',
        proximity_threshold=40,
        file_name_predict='tracking_and_predictions_DR.csv',
        file_name_alert='alert_times_DR.csv',
        target='person',
        source=0
    )
    tracker.run()
