import cv2
import numpy as np


import pygame

import utilitiesHelper  # Import utilities as helper functions
import time
from kalman import KalmanFilterWrapper
from ultralytics  import YOLO


class ObjectTracker_Kalman:

    def __init__(self, model_path, frequency, duration, proximity_threshold, file_name_predict, file_name_alert,
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
        self.duration = duration

    def run(self):
        """
        Main execution loop to read video frames and process detections continuously.
        """
        ret, frame = self.cap.read()  # Read once to get frame dimensions
        while ret:
            self.process_detection(utilitiesHelper.run_yolov8_inference(self.model, frame), frame)
            frame = utilitiesHelper.highlight_area(frame, self.any_area, self.label)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame = self.cap.read()

        utilitiesHelper.cleanup(self.cap, self.file)

    def trigger_proximity_alert(self, duration=2000, sound_file='awesomefollow.mp3'):
        """
        Triggers a sound to alert for proximity using a specified audio file.

        Parameters:
        - duration (int): Duration of the alert sound in milliseconds.
        - sound_file (str): Path to the sound file to play.
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

        Parameters:
        - detections (list): List of detected objects.
        - frame (np.array): Current video frame to process.
        """
        for det in detections:
            if utilitiesHelper.is_object_near_boundary(det, self.proximity_threshold, self.any_area):
                print("Alert: Object near the boundary detected!")
                break
            x1, y1, x2, y2, _, _, _ = det
            # Check if the detection is within the any_area and skip it
            if utilitiesHelper.is_area_excluded(x1, y1, x2, y2, self.any_area):
                continue
            color = utilitiesHelper.get_color_by_id(det[5])

            center_x, center_y, kf_wrapper = self.apply_kalman_filter(det) #For this detection
            future_x, future_y = kf_wrapper.predict() #Predict
            kf_wrapper.correct(np.array([[center_x], [center_y]])) #For update

            # Check if the detected object is NOT a person before logging
            #if det[6].lower() != 'person':
            utilitiesHelper.log_detection(self.writer, time.time(), center_x, center_y, future_x, future_y,
                                          det[6])  # Write to file
            utilitiesHelper.log_detection_data(det)
            utilitiesHelper.draw_predictions(frame, det, center_x, center_y, future_x, future_y, color) #draws those circles in middle
            if utilitiesHelper.is_object_near(det, self.any_area, self.proximity_threshold):
                pre_alert_time = time.time()
                #utilitiesHelper.trigger_proximity_alert(self.duration, self.frequency)
                #filePath = 'awesomefollow.mp3'
                #elf.trigger_proximity_alert(self.duration, filePath)
                post_alert_time = time.time()
                x1, y1, x2, y2, _, cls, class_name = det  # Get the topleft,x1,y1 and bottom_right x2,y2
                # Note: Passing self.start_time and self.center_area
                utilitiesHelper.handle_alert(self.alert_file, utilitiesHelper.save_alert_times, det, pre_alert_time,
                                             post_alert_time, x1, y1, x2, y2, self.start_time,
                                             self.any_area)

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
        #source=0,
        #source=r'bouncingbalLinear.mp4',
       source= r'bouncingbalDynamicParabel.mp4',
       # source=r'ball.mp4',
        duration=3000,
        frequency=2500,
        proximity_threshold=30,
        file_name_predict='tracking_and_predictions.csv',
        file_name_alert='alert_times.csv',
        label_name='Robotic Arm',
        any_area=((150, 150), (300, 300))
    )



    tracker.run()

    """
    Average Confidence Scores by Class:
    Class Name
    frisbee        0.443377
    sports ball    0.370734
    Name: Confidence Score, dtype: float64

    """
