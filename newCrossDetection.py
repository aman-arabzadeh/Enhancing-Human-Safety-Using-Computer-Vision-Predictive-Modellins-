import cv2
import numpy as np
import torch
import csv
import utilsNeeded
from ultralytics import YOLO

# Main Functionality

def main():
    # Load a pretrained YOLO model
    model = YOLO('yolov8n.pt')
    source = 0  # Video capture source
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)

    kalman_filters = {}

    # Open a file to save the detection and prediction data
    with open('tracking_and_predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['det_x', 'det_y', 'pred_x', 'pred_y', 'class_name'])  # Header

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO inference to get detections
            detections = utilsNeeded.run_yolov8_inference(model, frame)
            consolidated_detections = utilsNeeded.consolidate_detections(detections)

            # Perform Kalman filtering for object tracking
            for det in consolidated_detections:
                x1, y1, x2, y2, _, cls, class_name = det
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                if cls not in kalman_filters:
                    kalman_filters[cls] = cv2.KalmanFilter(4, 2)  # 4 dimensions (x, y, dx, dy), 2 measurements (x, y)
                    kalman_filters[cls].measurementMatrix = np.array([[1, 0, 0, 0],
                                                                       [0, 1, 0, 0]], np.float32)
                    kalman_filters[cls].transitionMatrix = np.array([[1, 0, 1, 0],
                                                                      [0, 1, 0, 1],
                                                                      [0, 0, 1, 0],
                                                                      [0, 0, 0, 1]], np.float32)
                    kalman_filters[cls].processNoiseCov = np.array([[1, 0, 0, 0],
                                                                     [0, 1, 0, 0],
                                                                     [0, 0, 1, 0],
                                                                     [0, 0, 0, 1]], np.float32) * 0.03
                kf = kalman_filters[cls]
                measurement = np.array([[center_x], [center_y]], np.float32)
                kf.correct(measurement)
                kf.predict()
                future_x, future_y = utilsNeeded.dead_reckoning2(kf,dt=1)

                # Save detection and prediction data
                writer.writerow([center_x, center_y, future_x, future_y, class_name])

                # Draw predicted position
                color = utilsNeeded.get_color_by_id(int(cls))
                cv2.circle(frame, (int(future_x), int(future_y)), 10, color, -1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{class_name} ({cls})"
                cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Filter detections to include person, cup, and other objects
            person_detections = [detection for detection in consolidated_detections if detection[6] == 'person']
            other_objects = [detection for detection in consolidated_detections if detection[6] != 'person']

            # Check proximity specifically for person detections
            if utilsNeeded.check_proximity(person_detections, other_objects):
               # print("Person detected near other objects.")
                utilsNeeded.beep_alert(frequency=3000, duration=500)

            # Display the resulting frame
            cv2.imshow("Cross Object Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
