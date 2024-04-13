import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import utilsNeeded
from ultralytics import YOLO

class ObjectTrackerApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0  # Video capture source
        self.model = YOLO('yolov8n.pt')
        self.kalman_filters = {}

        # Open video source
        self.vid = cv2.VideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Button to start/stop object tracking
        self.btn_track = tk.Button(window, text="Start Tracking", width=15, command=self.toggle_tracking)
        self.btn_track.pack(anchor=tk.CENTER, expand=True)

        # Bind the 'q' key to end the program
        self.window.bind('<Key>', self.key_pressed)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 10
        self.update()

        self.window.mainloop()

    def toggle_tracking(self):
        # Toggle object tracking on/off
        self.tracking = not getattr(self, 'tracking', False)
        if self.tracking:
            self.btn_track.config(text="Stop Tracking")
        else:
            self.btn_track.config(text="Start Tracking")

    def key_pressed(self, event):
        # End the program if the 'q' key is pressed
        if event.char == 'q':
            self.window.quit()

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()
        if ret:
            # Run YOLO inference to get detections
            detections = utilsNeeded.run_yolov8_inference(self.model, frame)

            # Perform Kalman filtering for object tracking
            for det in detections:
                _, _, _, _, _, cls, _ = det
                if cls not in self.kalman_filters:
                    self.kalman_filters[cls] = cv2.KalmanFilter(4, 2)  # 4 dimensions (x, y, dx, dy), 2 measurements (x, y)
                    self.kalman_filters[cls].measurementMatrix = np.array([[1, 0, 0, 0],
                                                                           [0, 1, 0, 0]], np.float32)
                    self.kalman_filters[cls].transitionMatrix = np.array([[1, 0, 1, 0],
                                                                          [0, 1, 0, 1],
                                                                          [0, 0, 1, 0],
                                                                          [0, 0, 0, 1]], np.float32)
                    self.kalman_filters[cls].processNoiseCov = np.array([[1, 0, 0, 0],
                                                                         [0, 1, 0, 0],
                                                                         [0, 0, 1, 0],
                                                                         [0, 0, 0, 1]], np.float32) * 0.03
                kf = self.kalman_filters[cls]
                x1, y1, x2, y2, _, _, _ = det
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                measurement = np.array([[center_x], [center_y]], np.float32)
                kf.correct(measurement)
                prediction = kf.predict()
                pred_x, pred_y = prediction[0], prediction[1]
                cv2.circle(frame, (int(pred_x), int(pred_y)), 10, (0, 255, 0), -1)

            # Display the resulting frame
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            # Update GUI
            self.window.after(self.delay, self.update)
        else:
            self.vid.release()

# Create a window and pass it to the Application object
ObjectTrackerApp(tk.Tk(), "Object Tracker")
