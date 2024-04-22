import cv2
import numpy as np
import uuid
import logging
import utilitiesHelper  # Import utilities as helper functions
import time
from kalman import KalmanFilterWrapper

class ObjectTracker_Kalman:
    def __init__(self, model_path, proximity_threshold, file_name_predict, file_name_alert, label_name, source=0, predefined_img_path=None):
        self.model = utilitiesHelper.load_model(model_path)
        self.cap = utilitiesHelper.initialize_video_capture(source)
        self.kalman_filters = {}
        self.file, self.writer = utilitiesHelper.setup_csv_writer(file_name_predict)
        self.alert_file = file_name_alert
        self.proximity_threshold = proximity_threshold
        self.predefined_image = cv2.imread(predefined_img_path) if predefined_img_path else None
        self.label = label_name
        self.center_area = None  # Will be updated upon the first frame read
        self.start_time = time.time()  # Set start_time

    def run(self):
        ret, frame = self.cap.read()  # Read once to get frame dimensions
        if ret:
            self.center_area = utilitiesHelper.update_center_area(frame.shape[1], frame.shape[0], factor=2)
            if self.predefined_image is not None:
                # Resize predefined image to fit center area dimensions
                area_width = self.center_area[1][0] - self.center_area[0][0]
                area_height = self.center_area[1][1] - self.center_area[0][1]
                self.predefined_image = cv2.resize(self.predefined_image, (area_width, area_height))

        while ret:
            self.process_detections(utilitiesHelper.run_yolov8_inference(self.model, frame), frame)
            frame = utilitiesHelper.highlight_center_area(frame, self.center_area, self.label, self.predefined_image)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame = self.cap.read()

        utilitiesHelper.cleanup(self.cap, self.file)

    def process_detections(self, detections, frame):
        for det in detections:
            object_id = str(uuid.uuid4())
            color = utilitiesHelper.get_color_by_id(object_id)
            center_x, center_y, kf_wrapper = self.apply_kalman_filter(det)
            future_x, future_y = kf_wrapper.predict()
            kf_wrapper.correct(np.array([[center_x], [center_y]]))
            utilitiesHelper.log_detection(self.writer, time.time(), center_x, center_y, future_x, future_y, det[6])
            utilitiesHelper.draw_predictions(frame, det, center_x, center_y, future_x, future_y, color)
            if utilitiesHelper.is_object_near(det, self.center_area, self.proximity_threshold):
                utilitiesHelper.trigger_proximity_alert(det)
                utilitiesHelper.handle_alert(self.alert_file, utilitiesHelper.save_alert_times, det, time.time(), center_x, center_y, future_x, future_y, self.start_time, self.center_area)

    def apply_kalman_filter(self, det):
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
        proximity_threshold=40,
        file_name_predict='tracking_and_predictions.csv',
        file_name_alert='alert_times.csv',
        predefined_img_path='../data/8.jpg',
        label_name='Robotic Arm'
    )
    tracker.run()
