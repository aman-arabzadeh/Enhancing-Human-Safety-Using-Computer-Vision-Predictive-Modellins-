import cv2
import numpy as np
import uuid
import csv
import logging
import utilsNeeded
import time
from kalmanSetUp import KalmanFilterWrapper
import hashlib
import winsound  # Ensure this module is available for playing beep sounds

def get_color_by_id(object_id):
    hash_code = int(hashlib.md5(object_id.encode()).hexdigest(), 16)
    return (hash_code & 255, (hash_code >> 8) & 255, (hash_code >> 16) & 255)

class ObjectTracker_Kalman:
    def __init__(self, model_path, proximity_threshold, file_name_predict, file_name_alert, source=0):
        self.model = utilsNeeded.load_model(model_path)
        self.cap = utilsNeeded.initialize_video_capture(source)
        self.kalman_filters = {}
        self.file, self.writer = utilsNeeded.setup_csv_writer(file_name_predict)
        self.start_time = time.time()
        self.proximity_threshold = proximity_threshold
        self.center_area = ((320, 240), (640, 480))  # Default center area

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to capture frame. Exiting...")
                break
            detections = utilsNeeded.run_yolov8_inference(self.model, frame)
            self.process_detections(detections, frame)
            self.highlight_center_area(frame)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        utilsNeeded.cleanup(self.cap, self.file)

    def check_proximity(self, det):
        # Simplified check for overlap
        (x1_obj, y1_obj, x2_obj, y2_obj, _, _, _) = det
        (top_left, bottom_right) = self.center_area
        x1_area, y1_area = top_left
        x2_area, y2_area = bottom_right
        return not (x2_obj < x1_area or x1_obj > x2_area or y2_obj < y1_area or y1_obj > y2_area)

    def check_nearness(self, det):
        # Simplified check for nearness without overlap
        (x1_obj, y1_obj, x2_obj, y2_obj, _, _, _) = det
        (top_left, bottom_right) = self.center_area
        x1_area, y1_area = top_left
        x2_area, y2_area = bottom_right
        return ((x1_obj > x2_area and (x1_obj - x2_area) <= self.proximity_threshold) or
                (x2_obj < x1_area and (x1_area - x2_obj) <= self.proximity_threshold) or
                (y1_obj > y2_area and (y1_obj - y2_area) <= self.proximity_threshold) or
                (y2_obj < y1_area and (y1_area - y2_obj) <= self.proximity_threshold))

    def highlight_center_area(self, frame):
        top_left, bottom_right = self.center_area
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # Part of ObjectTracker_Kalman class
    def process_detections(self, detections, frame):
        self.update_center_area(frame.shape[1], frame.shape[0])  # Update based on frame size
        for det in detections:
            object_id = str(uuid.uuid4())
            color = get_color_by_id(object_id)
            center_x, center_y, kf_wrapper = self.apply_kalman_filter(det)
            future_x, future_y = kf_wrapper.predict()
            kf_wrapper.correct(np.array([[center_x], [center_y]]))
            self.writer.writerow([time.time() - self.start_time, center_x, center_y, future_x, future_y, det[6]])
            utilsNeeded.draw_predictions2(frame, det, center_x, center_y, future_x, future_y, color)

            if self.check_proximity(det) or self.check_nearness(det):
                self.trigger_proximity_alert(det)

    def apply_kalman_filter(self, det):
        x1, y1, x2, y2, _, cls, class_name = det
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        if cls not in self.kalman_filters:
            self.kalman_filters[cls] = KalmanFilterWrapper()
            self.kalman_filters[cls].initialize(center_x, center_y)
        return center_x, center_y, self.kalman_filters[cls]

    def update_center_area(self, frame_width, frame_height):
        center_x, center_y = frame_width // 2, frame_height // 2
        area_width, area_height = frame_width // 3, frame_height // 3
        top_left = (center_x - area_width // 2, center_y - area_height // 2)
        bottom_right = (center_x + area_width // 2, center_y + area_height // 2)
        self.center_area = (top_left, bottom_right)



    def trigger_proximity_alert(self, det):
        print(f"Proximity alert: {det[6]} detected near or inside the central area!")
        winsound.Beep(2500, 1000)  # Beep sound for alert

if __name__ == "__main__":
    tracker = ObjectTracker_Kalman(
        'yolov8n.pt',
        source=0,
        proximity_threshold=20,
        file_name_predict='tracking_and_predictions.csv',
        file_name_alert='alert_times.csv'
    )
    tracker.run()
