import numpy as np
from kalman import KalmanFilterWrapper  # Ensure this is correctly imported


class TestKalman:
    def __init__(self):
        self.kalman_filters = {}
        self.last_coordinates = {}  # Stores the last coordinates for each class
        self.coordinate_threshold = 50  # Distance threshold to consider for reinitialization

    def is_significant_movement(self, cls, current_coords):
        """
        Determines if the detected object has moved significantly from its last known position.
        """
        if cls in self.last_coordinates:
            distance = np.linalg.norm(self.last_coordinates[cls] - current_coords)
            return distance > self.coordinate_threshold
        return True  # If no previous coordinates, assume significant movement

    def apply_kalman_filter(self, center_x, center_y, cls="test_object"):
        """
        Applies or initializes a Kalman filter for the detected object based on class and spatial consistency.
        """
        current_coords = np.array([center_x, center_y])
        if cls not in self.kalman_filters or self.is_significant_movement(cls, current_coords):
            self.kalman_filters[cls] = KalmanFilterWrapper()
            self.kalman_filters[cls].initialize(center_x, center_y)
        self.kalman_filters[cls].correct(np.array([[center_x], [center_y]]))
        predicted_x, predicted_y = self.kalman_filters[cls].predict()  # Ensure predict() method matches this usage
        self.last_coordinates[cls] = current_coords  # Update the last known coordinates
        return predicted_x, predicted_y

    def test_movements(self, test_coordinates):
        results = []
        for coords in test_coordinates:
            center_x, center_y = coords
            predicted_x, predicted_y = self.apply_kalman_filter(center_x, center_y)
            results.append((center_x, center_y, predicted_x, predicted_y))
        return results


# Generate test coordinates that incrementally simulate movement
test_coordinates = [(100 + i, 100 + int(i * 1.5)) for i in range(30)]

tester = TestKalman()
predictions = tester.test_movements(test_coordinates)

# Print the results
for i, (actual_x, actual_y, pred_x, pred_y) in enumerate(predictions):
    print(
        f"Test {i + 1}: Actual Position = ({actual_x}, {actual_y}), Predicted Next Position = ({pred_x:.2f}, {pred_y:.2f})")
