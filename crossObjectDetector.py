# main.py
import cv2
import numpy as np
import torch
import csv
import utilsNeeded
from ultralytics import YOLO



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
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')
    source = 0  # Video capture source
    cap = cv2.VideoCapture(source)
    kalman_filters = {}

    # Open a file to save the detection and prediction data for plotting , comparision for accurcy
    with open('tracking_and_predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['det_x', 'det_y', 'pred_x', 'pred_y', 'class_name'])  # Header

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = utilsNeeded.run_yolov8_inference(model, frame)

            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf, cls, class_name = det
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                if cls not in kalman_filters:
                    kalman_filters[cls] = utilsNeeded.KalmanFilter(dt=0.1, u=0.0, std_acc=1, std_meas=0.5)

                kf = kalman_filters[cls]
                meas = np.array([[center_x], [center_y]])
                kf.update(meas)
                kf.predict()

                future_x, future_y = utilsNeeded.dead_reckoning(kf, dt=1)

                # Save detection and prediction data
                writer.writerow([center_x, center_y, future_x, future_y, class_name])

                color = utilsNeeded.get_color_by_id(int(cls))
                cv2.circle(frame, (future_x, future_y), 10, color, -1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

          #  if check_proximity(detections):
           #    # Play beep sound with custom frequency and duration
            # beep_alert(frequency=3000, duration=500)

            cv2.imshow("Cross Object Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): #https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    main()

# git status
#git push origin main
