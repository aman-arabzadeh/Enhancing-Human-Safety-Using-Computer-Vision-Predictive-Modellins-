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
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.initialized = False

    def initialize(self, initial_x, initial_y, initial_dx=0, initial_dy=0):
        """Initialize the filter with the first detected position and estimated velocity."""
        self.kf.statePost = np.array([[initial_x], [initial_y], [initial_dx], [initial_dy]], dtype=np.float32)
        self.initialized = True
        logging.info(
            f"Kalman filter initialized with position ({initial_x}, {initial_y}) and velocity ({initial_dx}, {initial_dy}).")

    def correct(self, measurement):
        """Correct the Kalman filter with the detected measurement."""
        if not self.initialized:
            logging.error("Kalman filter must be initialized before correction.")
            return None  # Just return None or handle appropriately
        corrected = self.kf.correct(np.array(measurement, dtype=np.float32))
        return corrected

    def predict(self):
        """Predict the next state of the object using the Kalman filter."""
        if not self.initialized:
            logging.error("Kalman filter must be initialized before prediction.")
            return None
        prediction = self.kf.predict()
        self.future_x, self.future_y = int(prediction[0]), int(prediction[1])
        logging.info(f"Predicted future position: ({self.future_x}, {self.future_y})")
        return self.future_x, self.future_y
