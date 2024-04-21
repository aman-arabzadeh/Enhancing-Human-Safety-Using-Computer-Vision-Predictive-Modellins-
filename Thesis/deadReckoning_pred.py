import csv
import os
import cv2
import numpy as np
import uuid
import logging
import utilitiesHelper  # Import utilities as helper functions
import time


class DeadReckoningTracker:
    def __init__(self, model_path, proximity_threshold, file_name_predict, file_name_alert, source=0):
        self.model = utilitiesHelper.load_model(model_path)
        self.cap = utilitiesHelper.initialize_video_capture(source)
        self.file, self.writer = utilitiesHelper.setup_csv_writer(file_name_predict)
        self.alert_file = file_name_alert
        self.start_time = time.time()
        self.proximity_threshold = proximity_threshold
        self.last_positions = {}
        self.entry_times = {}
        self.alert_times = []
        self.alert_start_time = None
        self.center_area = None  # Will be updated upon the first frame read



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

    def process_detections(self, detections, frame):
        timestamp = time.time()
        for det in detections:
            object_id = str(uuid.uuid4())  # Consider tracking objects across frames if relevant
            color = utilitiesHelper.get_color_by_id(object_id)
            center_x, center_y, future_x, future_y = self.apply_dead_reckoning(det, timestamp)
            utilitiesHelper.log_detection(self.writer, timestamp, center_x, center_y, future_x, future_y, det[6])
            utilitiesHelper.draw_predictions(frame, det, center_x, center_y, future_x, future_y, color)
            if utilitiesHelper.is_object_near(det, self.center_area, self.proximity_threshold):
                utilitiesHelper.trigger_proximity_alert(det)
                utilitiesHelper.handle_alert(self.alert_file, utilitiesHelper.save_alert_times, det, timestamp, center_x, center_y, future_x, future_y,
                             self.start_time, self.center_area)

    def run(self):
        ret, frame = self.cap.read()  # Read once to get frame dimensions
        if ret:
            self.center_area = utilitiesHelper.update_center_area(frame.shape[1], frame.shape[0], factor=2)

        while ret:
            detections = utilitiesHelper.run_yolov8_inference(self.model, frame)
            self.process_detections(detections, frame)
            utilitiesHelper.highlight_center_area(frame, self.center_area)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame = self.cap.read()

        utilitiesHelper.cleanup(self.cap, self.file)



if __name__ == "__main__":
    tracker = DeadReckoningTracker(
        model_path='yolov8n.pt',
        proximity_threshold=40,
        file_name_predict='tracking_and_predictions_DR.csv',
        file_name_alert='alert_times_DR.csv',
        source=0
    )
    tracker.run()
