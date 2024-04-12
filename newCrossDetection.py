import cv2
import numpy as np
import utilsNeeded
from ultralytics import YOLO

def create_kalman_filter():
    kf = cv2.KalmanFilter(4, 2)  # 4 dynamic parameters (x, y, dx, dy), 2 measured parameters
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
    kf.statePost = np.zeros(4, dtype=np.float32)
    return kf

def update_kalman_filter(kf, measurement):
    measurement = np.array(measurement, dtype=np.float32).reshape(2, 1)
    kf.correct(measurement)
    predicted = kf.predict()
    return predicted

def main():
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
    print("Frame Rate:", fps)

    kalman_filters = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        detections = utilsNeeded.run_yolov8_inference(model, frame)
        for det in detections:
            x1, y1, x2, y2, conf, cls, class_name = det
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

            if cls not in kalman_filters:
                kf = create_kalman_filter()
                kf.statePost[:2] = np.array([center_x, center_y], dtype=np.float32)
                kalman_filters[cls] = kf
            else:
                kf = kalman_filters[cls]

            predicted = update_kalman_filter(kf, [center_x, center_y])
            print("Predicted state:", predicted.ravel())

            # Visualize the bounding box and prediction
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            pred_x, pred_y = int(predicted[0]), int(predicted[1])
            cv2.circle(frame, (pred_x, pred_y), 5, (0, 255, 0), -1)  # Green dot for prediction

        # Show the frame
        cv2.imshow("Cross Object Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
