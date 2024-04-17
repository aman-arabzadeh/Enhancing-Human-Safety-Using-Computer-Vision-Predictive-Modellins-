import cv2
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KalmanFilterWrapper:
    def __init__(self):
        self.future_x = None
        self.future_y = None
        self.kf = cv2.KalmanFilter(4, 2)  # State: [x, y, dx, dy], Measurement: [x, y]
        self.kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.base_process_noise = np.eye(4, dtype=np.float32) * 0.03
        self.kf.processNoiseCov = self.base_process_noise
        self.initialized = False

    def initialize(self, initial_x, initial_y, initial_dx=0, initial_dy=0):
        """Initialize the filter with the first detected position and estimated velocity."""
        self.kf.statePost = np.array([[initial_x], [initial_y], [initial_dx], [initial_dy]], dtype=np.float32)
        self.initialized = True
        logging.info(f"Kalman filter initialized with position ({initial_x}, {initial_y}) and velocity ({initial_dx}, {initial_dy}).")

    def correct(self, measurement):
        """Correct the Kalman filter with the detected measurement."""
        if not self.initialized:
            logging.error("Kalman filter must be initialized before correction.")
            return
        self.kf.correct(np.array(measurement, dtype=np.float32))

    def predict(self, fps, velocity_scale=1.0):
        """Predict the next state of the object using the Kalman filter."""
        if not self.initialized:
            logging.error("Kalman filter must be initialized before prediction.")
            return
        self.kf.processNoiseCov = self.base_process_noise * velocity_scale
        prediction = self.kf.predict()
        current_predicted_x = prediction[0, 0]
        current_predicted_y = prediction[1, 0]
        velocity_x = prediction[2, 0]
        velocity_y = prediction[3, 0]
        dt = 1 / fps
        self.future_x = current_predicted_x + velocity_x * dt
        self.future_y = current_predicted_y + velocity_y * dt
        return self.future_x, self.future_y
