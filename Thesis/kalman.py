import cv2
import numpy as np
import logging
# Authorship Information
"""
Author: Koray Aman Arabzadeh
Thesis: Mid Sweden University.
Bachelor Thesis - Bachelor of Science in Engineering, Specialisation in Computer Engineering
Main field of study: Computer Engineering
Credits: 15 hp (ECTS)
Semester, Year: Spring, 2024
Supervisor: Emin Zerman
Examiner: Stefan Forsström
Course code: DT099G
Programme: Degree of Bachelor of Science with a major in Computer Engineering

"""

# Setup logging to display information with a timestamp, log level, and message.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# External references for further information and context:
# Kalman Filter example from GitHub: https://github.com/SriramEmarose/Motion-Prediction-with-Kalman-Filter/blob/master/KalmanFilter.py
# YouTube videos for understanding dynamic and linear motion capture:
# Dynamic Motion: https://www.youtube.com/watch?v=sKJegbjS4N8
# Linear Motion: https://www.youtube.com/watch?v=zsdPYFPTdw0

class KalmanFilterWrapper:
    """
    A wrapper class for OpenCV's Kalman Filter, tailored for tracking objects in video streams.
    This class manages the state of an object in terms of both position (x, y) and velocity (dx, dy).
    """
    def __init__(self):
        """
        Initializes the Kalman Filter with the appropriate state and measurement dimensions.
        The state vector includes position and velocity: [x, y, dx, dy].
        The measurement vector includes position only: [x, y].
        """
        # Create a Kalman Filter with 4 state dimensions (position and velocity) and 2 measurement dimensions (position).
        self.kf = cv2.KalmanFilter(4, 2)
        # Define how measurements map to the state vector.
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        # Define the state transition model, which describes how the state evolves from one timestep to the next.
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], dtype=np.float32)
        # Flag to check if the filter is initialized with the first set of measurements.
        self.initialized = False

    def initialize(self, initial_x, initial_y, initial_dx=0, initial_dy=0):
        """
        Set the initial state of the Kalman filter.
        Args:
            initial_x (float): Initial x position.
            initial_y (float): Initial y position.
            initial_dx (float): Initial x velocity (optional, default 0).
            initial_dy (float): Initial y velocity (optional, default 0).
        """
        # Set the initial state with position and velocity.
        self.kf.statePost = np.array([[initial_x], [initial_y], [initial_dx], [initial_dy]], dtype=np.float32)
        self.initialized = True
        logging.info(f"Kalman filter initialized with position ({initial_x}, {initial_y}) and velocity ({initial_dx}, {initial_dy}).")

    def correct(self, measurement):
        """
        Update the Kalman filter with a new measurement.
        Args:
            measurement (list): The measured [x, y] position of the object.
        Returns:
            numpy.ndarray: The corrected state estimate after incorporating the measurement.
        """
        # Ensure the filter is initialized before proceeding with correction.
        if not self.initialized:
            logging.error("Kalman filter must be initialized before correction.")
            return None
        # Apply the correction using the new measurement.
        corrected = self.kf.correct(np.array(measurement, dtype=np.float32))
        return corrected

    def predict(self):
        """
        Predict the next state of the object using the current state estimate.
        Returns:
            tuple: Predicted future position (x, y) based on the model.
        """
        # Ensure the filter is initialized before proceeding with prediction.
        if not self.initialized:
            logging.error("Kalman filter must be initialized before prediction.")
            return None
        # Calculate the predicted next state.
        prediction = self.kf.predict()
        # Extract the predicted position from the state vector.
        self.future_x, self.future_y = int(prediction[0]), int(prediction[1])
        logging.info(f"Predicted future position: ({self.future_x}, {self.future_y})")
        return self.future_x, self.future_y
