import cv2
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KalmanFilterWrapper:
    """
    A wrapper class for OpenCV's Kalman Filter, tailored for tracking objects in video streams.
    The state consists of position and velocity in the x and y directions.
    """
    def __init__(self):
        """
        Initializes the Kalman Filter with the state and measurement dimensions.
        The state vector is [x, y, dx, dy] and the measurement vector is [x, y].
        """
        self.future_x = None
        self.future_y = None
        self.kf = cv2.KalmanFilter(4, 2)  # 4 state variables, 2 measurement variables
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], dtype=np.float32)
        self.initialized = False

    def initialize(self, initial_x, initial_y, initial_dx=0, initial_dy=0):
        """
        Initialize the filter with the first detected position and an estimated initial velocity.
        Parameters:
            initial_x (float): Initial x position.
            initial_y (float): Initial y position.
            initial_dx (float): Initial x velocity (default 0).
            initial_dy (float): Initial y velocity (default 0).
        """
        self.kf.statePost = np.array([[initial_x], [initial_y], [initial_dx], [initial_dy]], dtype=np.float32)
        self.initialized = True
        logging.info(f"Kalman filter initialized with position ({initial_x}, {initial_y}) and velocity ({initial_dx}, {initial_dy}).")

    def correct(self, measurement):
        """
        Correct the Kalman filter with the detected measurement.
        Parameters:
            measurement (list): The measured [x, y] position of the object.
        Returns:
            numpy.ndarray: The corrected state estimate.
        """
        if not self.initialized:
            logging.error("Kalman filter must be initialized before correction.")
            return None
        corrected = self.kf.correct(np.array(measurement, dtype=np.float32))
        return corrected

    def predict(self):
        """
        Predict the next state of the object using the Kalman filter.
        Returns:
            tuple: Predicted future position (x, y).
        """
        if not self.initialized:
            logging.error("Kalman filter must be initialized before prediction.")
            return None
        prediction = self.kf.predict()
        self.future_x, self.future_y = int(prediction[0]), int(prediction[1])
        logging.info(f"Predicted future position: ({self.future_x}, {self.future_y})")
        return self.future_x, self.future_y
