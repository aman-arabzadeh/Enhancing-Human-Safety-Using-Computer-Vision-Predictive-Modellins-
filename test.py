
from ultralytics import YOLO # Import the library

model = YOLO("yolov8n.pt")  # load a pretrained model
# Use the model

results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

