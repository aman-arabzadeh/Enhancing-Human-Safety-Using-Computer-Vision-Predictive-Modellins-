import numpy as np
import cv2

#  Testing a basic Kalman Filter for a 2D (x, y) position tracking scenario, by openCV
# State Vector [x, y, dx, dy] - position and velocity
# Measurement Vector [x, y] - position

# Initialize the Kalman Filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  # Process noise
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.001  # Measurement noise
kalman.errorCovPost = np.eye(4, dtype=np.float32)

# Initial State
initial_state = np.array([0, 0, 0, 0], np.float32)
kalman.statePost = initial_state

# Dummy variable for this example
measurements = np.array([[1, 2], [2, 4], [3, 6], [4, 8]], np.float32)  # Example measurements

for measurement in measurements:
    # Prediction
    predicted = kalman.predict()
    print(f"Predicted state: {predicted[:2]}")

    # Updating with measurement
    corrected = kalman.correct(measurement)
    print(f"Updated state: {corrected[:2]}")
