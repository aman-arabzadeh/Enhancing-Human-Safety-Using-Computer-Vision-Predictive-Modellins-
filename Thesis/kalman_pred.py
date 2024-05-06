import cv2
import numpy as np
import pygame
import utilitiesHelper  # Import utilities as helper functions
import time
from kalman import KalmanFilterWrapper

class ObjectTracker_Kalman:
    def __init__(self, model_path, frequency, duration, proximity_threshold, file_name_predict, file_name_alert,coordinate_threshold,
                 label_name, source=0, any_area=None):
        """
        Initializes the object tracker with specified parameters and manually set any_area.
        """
        self.model = utilitiesHelper.load_model(model_path)
        self.cap = utilitiesHelper.initialize_video_capture(source)
        self.kalman_filters = {}
        self.file, self.writer = utilitiesHelper.setup_csv_writer(file_name_predict)
        self.alert_file = file_name_alert
        self.proximity_threshold = proximity_threshold
        self.label = label_name
        self.any_area = any_area  # Manually set as ((x1, y1), (x2, y2))
        self.start_time = time.time()
        self.frequency = frequency
        self.kalman_filters = {}
        self.last_coordinates = {}  # Stores the last coordinates for each class
        self.coordinate_threshold = coordinate_threshold  # Distance threshold to consider for reinitialization
        self.duration = duration
        self.classNames = ["person", "sports ball", "cup", "chair"]

    def run(self):
        """
        Main execution loop to read video frames and process detections continuously.
        """
        ret, frame = self.cap.read()
        while ret:
            detections = utilitiesHelper.run_yolov8_inference(self.model, frame)
            self.process_detection(detections, frame)
            frame = utilitiesHelper.highlight_area(frame, self.any_area, self.label)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame = self.cap.read()
        utilitiesHelper.cleanup(self.cap, self.file)

    def trigger_proximity_alert(self, duration=2000, sound_file='awesomefollow.mp3'):
        """
        Triggers a sound to alert for proximity using a specified audio file.
        """
        pygame.mixer.init()
        try:
            sound = pygame.mixer.Sound(sound_file)
            sound.play()
            pygame.time.delay(duration)
            sound.stop()
        except Exception as e:
            print(f"Error playing sound: {str(e)}")
        finally:
            pygame.mixer.quit()



    def process_detection(self, detections, frame):
        """
        Processes detected objects, applies Kalman filters, logs detections, and checks for proximity hazards.
        Also checks for significant overlaps between detections to handle object identity management.
        """
        for det in detections:
            x1, y1, x2, y2, _, cls, class_name = det
            #if class_name not in self.classNames:
             #   continue
            if utilitiesHelper.is_area_excluded(x1, y1, x2, y2,self.any_area):
                continue
            self.manage_detections(det, frame)


    def manage_detections(self, det, frame):
        """
        Apply Kalman filter, log, draw, and handle proximity alerts for new and non-overlapping detections.
        """
        x1, y1, x2, y2, _, cls, class_name = det
        color = utilitiesHelper.get_color_by_id(cls)
        center_x, center_y, kf_wrapper = self.apply_kalman_filter(det)
        future_x, future_y = kf_wrapper.predict()
        kf_wrapper.correct(np.array([[center_x], [center_y]]))
        if class_name.lower() != 'person':
            utilitiesHelper.log_detection(self.writer, time.time(), center_x, center_y, future_x, future_y, class_name)
            utilitiesHelper.log_detection_data(det)
        utilitiesHelper.draw_predictions(frame, det, center_x, center_y, future_x, future_y, color)
        #if utilitiesHelper.is_object_near(det, self.any_area, self.proximity_threshold):
            #self.handle_proximity_alert(det, x1, y1, x2, y2)

    def is_significant_movement(self, cls, current_coords):
        if cls in self.last_coordinates:
            distance = np.linalg.norm(self.last_coordinates[cls] - current_coords) #Checks eucleadean distance
            print(distance, self.last_coordinates[cls], current_coords)
            return distance > self.coordinate_threshold
        return True  # Assume significant movement if no previous coordinates

    def apply_kalman_filter(self, det):
        x1, y1, x2, y2, _, cls, class_name = det
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        current_coords = np.array([center_x, center_y])
        # suggest that the Kalman filter should continue with its current state without reinitialization, assuming it already exists for this object.
        if cls not in self.kalman_filters or self.is_significant_movement(cls, current_coords):
            self.kalman_filters[cls] = KalmanFilterWrapper()
            self.kalman_filters[cls].initialize(center_x, center_y)

        self.last_coordinates[cls] = current_coords  # Update the last known coordinates

        return center_x, center_y, self.kalman_filters[cls]


    def handle_proximity_alert(self, det, x1, y1, x2, y2):
        """
        Handle proximity alerts for detected objects near any_area.
        """
        pre_alert_time = time.time()
        utilitiesHelper.trigger_proximity_alert(self.duration, self.frequency)
        post_alert_time = time.time()
        utilitiesHelper.handle_alert(self.alert_file, utilitiesHelper.save_alert_times, det, pre_alert_time,
                                     post_alert_time, x1, y1, x2, y2, self.start_time, self.any_area)

if __name__ == "__main__":
    tracker = ObjectTracker_Kalman(
        'yolov8n.pt',
        #source=0,
        #source=r'apple.mp4',
        source=r'appleParabolic.mp4',
        coordinate_threshold = 20,
        duration=3000,
        frequency=2500,
        proximity_threshold=30,
        file_name_predict='tracking_and_predictions.csv',
        file_name_alert='alert_times.csv',
        label_name='Robotic Arm',
        any_area=((150, 150), (300, 300))
    )
    tracker.run()
