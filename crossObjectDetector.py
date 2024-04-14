import cv2
import numpy as np
import torch
import csv
import utilsNeeded
from ultralytics import YOLO
from KalmanFilter import KalmanFilter
import time  # Import time module to work with timestamps

def main():
    # Load a pretrained YOLO model
    model = YOLO('yolov8n.pt')
    source = 0  # Video capture source
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_time = time.time()  # Capture start time

    kalman_filters = {}

    # Open a file to save the detection and prediction data
    with open('tracking_and_predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'det_x', 'det_y', 'pred_x', 'pred_y', 'class_name'])  # Updated header

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO inference to get detections
            detections = utilsNeeded.run_yolov8_inference(model, frame)
            consolidated_detections = utilsNeeded.consolidate_detections(detections)

            for det in consolidated_detections:
                x1, y1, x2, y2, conf, cls, class_name = det
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                if cls not in kalman_filters:
                    kalman_filters[cls] = KalmanFilter(dt=1/fps, u=0.0, std_acc=0.1, std_meas=0.1)
                kf = kalman_filters[cls]
                kf.predict()
                meas = np.array([[center_x], [center_y]])
                kf.update(meas)

                future_x, future_y = utilsNeeded.dead_reckoning(kf, dt=1)

                # Include timestamp in saved data
                current_time = time.time() - start_time  # Elapsed time since start
                writer.writerow([current_time, center_x, center_y, future_x, future_y, class_name])

                # Visualization code unchanged
                color = utilsNeeded.get_color_by_id(int(cls))
                cv2.circle(frame, (future_x, future_y), 10, color, -1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{class_name} ({cls}): {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display and quit condition
            cv2.imshow("Cross Object Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
