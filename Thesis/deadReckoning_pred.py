import cv2
import utilitiesHelper  # Helper utilities for model loading, video capture, etc.
import time

class DeadReckoningTracker:
    """
    A tracker that uses dead reckoning to predict future positions of objects based on their velocities
    and past positions. The tracker is integrated with object detection to update and predict object positions
    in video streams.

    Attributes:
        model_path (str): Path to the object detection model.
        frequency (int): Frequency of alert sound.
        duration (int): Duration of alert sound.
        factor (int): Scaling factor for calculating the center area in the video.
        proximity_threshold (int): Distance threshold for triggering proximity alerts.
        file_name_predict (str): File name for logging predictions.
        file_name_alert (str): File name for logging alert times.
        label_name (str): Label for detection.
        source (int, optional): Video source index or path.
        predefined_img_path (str, optional): Path to an image for overlay purposes.
    """
    def __init__(self, model_path, frequency, duration, factor, proximity_threshold, file_name_predict, file_name_alert, label_name, source=0, predefined_img_path=None,any_area=None):
        self.model = utilitiesHelper.load_model(model_path)
        self.cap = utilitiesHelper.initialize_video_capture(source)
        self.factor = factor
        self.file, self.writer = utilitiesHelper.setup_csv_writer(file_name_predict)
        self.alert_file = file_name_alert
        self.proximity_threshold = proximity_threshold
        self.last_positions = {}
        self.alert_times = []
        self.label = label_name
        self.start_time = time.time()
        self.predefined_image = cv2.imread(predefined_img_path) if predefined_img_path else None
        self.frequency = frequency
        self.duration = duration
        self.any_area = any_area

    def run(self):
        """
        Executes the main tracking loop, capturing frames and processing detections until termination.
        Uses a predefined image to overlay if provided and processes each frame until the 'q' key is pressed.
        """
        ret, frame = self.cap.read()  # Initial read to get frame dimensions
        while ret:
            detections = utilitiesHelper.run_yolov8_inference(self.model, frame)
            self.process_detection(detections, frame)
            frame = utilitiesHelper.highlight_area(frame, self.any_area, self.label)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame = self.cap.read()

        utilitiesHelper.cleanup(self.cap, self.file)

    def process_detection(self, detections, frame):
            """
            Processes each detection from a list of detections provided by the object detection model,
            applies necessary logic, and checks for proximity hazards.

            Parameters:
            - detections (list of tuples): List of detection information from the object detection model.
            - frame (np.array): The current frame being processed.
            """
            for det in detections:
                x1, y1, x2, y2, _, object_id, class_name = det  # Assuming detections are structured this way

                # If the detected object is within the 'any_area', skip further processing
                if x1 >= self.any_area[0][0] and x2 <= self.any_area[1][0] and \
                        y1 >= self.any_area[0][1] and y2 <= self.any_area[1][1]:
                    continue  # Skip processing for this detection

                # Check proximity to boundaries
                if utilitiesHelper.is_object_near_boundary(det, 10, self.any_area):
                    print("Alert: Object near the boundary detected!")

                color = utilitiesHelper.get_color_by_id(object_id)
                center_x, center_y, future_x, future_y = self.apply_dead_reckoning(det, time.time())

                # Log the detection if it is not a person
                if class_name.lower() != 'person':
                    utilitiesHelper.log_detection(self.writer, time.time(), center_x, center_y, future_x, future_y,
                                                  class_name)
                    utilitiesHelper.log_detection_data(det)

                # Draw predictions on the frame
                utilitiesHelper.draw_predictions(frame, det, center_x, center_y, future_x, future_y, color)

                # Check proximity to the specified area and trigger alerts if necessary
                if utilitiesHelper.is_object_near(det, self.any_area, self.proximity_threshold):
                    pre_alert_time = time.time()
                    # Placeholder for alert triggering function
                    post_alert_time = time.time()
                    utilitiesHelper.trigger_proximity_alert(self.duration, self.frequency)
                    utilitiesHelper.handle_alert(self.alert_file, utilitiesHelper.save_alert_times, det, pre_alert_time,
                                                 post_alert_time, center_x, center_y, future_x, future_y,
                                                 self.start_time,
                                                 self.any_area)

    def apply_dead_reckoning(self, det, timestamp):
        """
        Applies dead reckoning to predict the future position of an object based on its current and last known positions.

        Parameters:
        - det (list): The detection data containing bounding box and class information.
        - timestamp (float): The current time used for calculating movement speed.

        Returns:
        - tuple: Current and predicted future positions of the object.
        """
        x1, y1, x2, y2, _, cls, _ = det
        current_x = int((x1 + x2) / 2)
        current_y = int((y1 + y2) / 2)
        last_info = self.last_positions.get(cls, (current_x, current_y, timestamp))
        time_delta = timestamp - last_info[2]
        velocity_x = (current_x - last_info[0]) / time_delta if time_delta > 0 else 0
        velocity_y = (current_y - last_info[1]) / time_delta if time_delta > 0 else 0
        future_x = int(current_x + velocity_x * time_delta)
        future_y = int(current_y + velocity_y * time_delta)
        self.last_positions[cls] = (current_x, current_y, timestamp)
        return current_x, current_y, future_x, future_y


if __name__ == "__main__":
    tracker = DeadReckoningTracker(
        'yolov8n.pt',
        #source=0,
        source=r'bouncingbalLinear.mp4',
        #source= r'bouncingbalDynamicParabel.mp4',
        duration=1000,
        frequency=2500,
        proximity_threshold=70,
        factor=4,
        file_name_predict='tracking_and_predictions_DR.csv',
        file_name_alert='alert_times_DR.csv',
        predefined_img_path='../data/8.jpg',
        label_name="robotic arm",
        any_area=((100, 100), (350, 350))

    )
    tracker.run()
