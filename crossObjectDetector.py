# main.py
import cv2
import numpy as np
import torch
import csv
import utilsNeeded
from ultralytics import YOLO
from KalmanFilter import KalmanFilter

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



Resources used: 
https://opencv.org/
https://stackoverflow.com/
https://github.com
https://pieriantraining.com/kalman-filter-opencv-python-example/
"""

# Main Functionality
"""
This Python script integrates YOLOv5 for object detection, Kalman filtering for object tracking,
dead reckoning for predicting future positions, and audio alerts for detecting close object proximity.
This code can monitor movements, detect anomalies or collisions, and trigger alerts for maintenance or safety measures.
"""



# Usage Instructions
"""
To run, use the following command:
python .\crossObjectDetector.py  

This code can be used for navigating dynamic environments, like warehouses, to track objects or other robots, plan paths, and avoid collisions.
"""
def main():
    # Load a pretrained YOLO model
    model = YOLO('yolov8n.pt')
    source = 0  # Video capture source
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frame Rate:", fps)

    kalman_filters = {}
  #  object_id_counter = 1  # Initialize object ID counter

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

            # Consolidate detections to prevent duplicate tracking
            consolidated_detections = utilsNeeded.consolidate_detections(detections)

            for det in consolidated_detections:
                x1, y1, x2, y2, conf, cls, class_name = det
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                #object_id = object_id_counter
               # object_id_counter += 1

                #if object_id not in kalman_filters:
                kalman_filters[cls] = KalmanFilter(dt=1 / fps, u=0.0, std_acc=0.1, std_meas=0.1)

                kf = kalman_filters[cls]
                kf.predict()
                meas = np.array([[center_x], [center_y]])
                kf.update(meas)

                future_x, future_y = utilsNeeded.dead_reckoning(kf, dt=1)

                # Save detection and prediction data
                writer.writerow([center_x, center_y, future_x, future_y, class_name])

                color = utilsNeeded.get_color_by_id(int(cls))
                cv2.circle(frame, (future_x, future_y), 10, color, -1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{class_name} ({cls}): {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Filter detections to include only person
            person_detections = [detection for detection in consolidated_detections if detection[6] == 'person']

            # Check proximity specifically for person detections
            if utilsNeeded.check_proximity(person_detections):
                print("person in close proximity detected.")
                utilsNeeded.beep_alert(frequency=3000, duration=500)

            cv2.imshow("Cross Object Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
