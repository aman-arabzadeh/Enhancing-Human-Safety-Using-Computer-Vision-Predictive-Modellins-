import cv2
import numpy as np
import uuid
import logging
import utilitiesHelper  # Import utilities as helper functions
import time
from kalman import KalmanFilterWrapper


class ObjectTracker_Kalman:
    """
    A class to track objects using the Kalman filter in real-time video feeds with the ability to detect,
    log, and alert on proximity hazards around a robotic arm.

    Attributes:
    - model (YOLO): The object detection model loaded for identifying objects.
    - cap (cv2.VideoCapture): Video capture object for frame acquisition.
    - kalman_filters (dict): Dictionary storing Kalman filters for each tracked class.
    - file (file object): File object for logging detection data.
    - writer (csv.writer): CSV writer object for writing detection data.
    - alert_file (str): Path to the alert log file.
    - proximity_threshold (int): Distance threshold for proximity alerts.
    - predefined_image (np.array): Image to overlay on detection areas.
    - label (str): Label for the detection area.
    - center_area (tuple): Coordinates for the central area of interest in the video frame.
    - start_time (float): Timestamp when tracking started.
    - frequency (int): Frequency of the proximity alert sound.
    - duration (int): Duration of the proximity alert sound.

    Methods:
    - run(): Main loop to capture video frames and process detections.
    - process_detections(detections, frame): Process each detection per frame.
    - apply_kalman_filter(det): Apply or initialize a Kalman filter based on detection.
    """

    def __init__(self, model_path, frequency, duration, proximity_threshold, file_name_predict, file_name_alert,
                 label_name, source=0, predefined_img_path=None):
        """
        Initializes the object tracker with specified parameters and setups necessary utilities.

        Parameters:
        - model_path (str): Path to the object detection model file.
        - frequency (int): Frequency of the alert sound.
        - duration (int): Duration of the alert sound.
        - proximity_threshold (int): Proximity threshold for triggering alerts.
        - file_name_predict (str): Path to the CSV file for prediction logging.
        - file_name_alert (str): Path to the CSV file for alert logging.
        - label_name (str): Label for the detected objects.
        - source (int, optional): Video source index or path. Defaults to 0.
        - predefined_img_path (str, optional): Path to an image for overlay purposes. Defaults to None.
        """
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
        self.frequency = frequency
        self.duration = duration

    def run(self):
        """
        Main execution loop to read video frames and process detections continuously.
        """
        ret, frame = self.cap.read()  # Read once to get frame dimensions
        if ret:
            self.center_area = utilitiesHelper.update_center_area(frame.shape[1], frame.shape[0], factor=2)
            if self.predefined_image is not None:
                area_width = self.center_area[1][0] - self.center_area[0][0]
                area_height = self.center_area[1][1] - self.center_area[0][1]
                self.predefined_image = cv2.resize(self.predefined_image, (area_width, area_height))

        while ret:
            self.process_detection(utilitiesHelper.run_yolov8_inference(self.model, frame), frame)
            frame = utilitiesHelper.highlight_center_area(frame, self.center_area, self.label, self.predefined_image)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame = self.cap.read()

        utilitiesHelper.cleanup(self.cap, self.file)

    def process_detection(self, detections, frame):
        """
        Processes detected objects, applies Kalman filters, logs detections, and checks for proximity hazards.

        Parameters:
        - detections (list): List of detected objects.
        - frame (np.array): Current video frame to process.
        """
        for det in detections:
            object_id = str(uuid.uuid4())
            color = utilitiesHelper.get_color_by_id(object_id)
            center_x, center_y, kf_wrapper = self.apply_kalman_filter(det)
            future_x, future_y = kf_wrapper.predict()
            kf_wrapper.correct(np.array([[center_x], [center_y]]))
            utilitiesHelper.log_detection(self.writer, time.time(), center_x, center_y, future_x, future_y, det[6])
            utilitiesHelper.draw_predictions(frame, det, center_x, center_y, future_x, future_y, color)
            if utilitiesHelper.is_object_near(det, self.center_area, self.proximity_threshold):
                print(f"Proximity alert: {det[6]} detected near or inside the central area!")
                utilitiesHelper.trigger_proximity_alert(self.duration, self.frequency)
                utilitiesHelper.handle_alert(self.alert_file, utilitiesHelper.save_alert_times, det, time.time(),
                                             center_x, center_y, future_x, future_y, self.start_time, self.center_area)

    def apply_kalman_filter(self, det):
        """
        Applies or initializes a Kalman filter for the detected object.

        Parameters:
        - det (list): Detection data of the object.

        Returns:
        - tuple: Current center coordinates of the object and its Kalman filter wrapper.
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
        duration=1000,
        frequency=2500,
        source=0,
        proximity_threshold=40,
        file_name_predict='tracking_and_predictions.csv',
        file_name_alert='alert_times.csv',
        predefined_img_path='../data/8.jpg',
        label_name='Robotic Arm'
    )
    tracker.run()
