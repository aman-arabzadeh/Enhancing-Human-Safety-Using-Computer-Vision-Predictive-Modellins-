from ultralytics import YOLO

# Load the model
model = YOLO('yolov8n')

# Run detection
image_path = r'C:\Users\amana\PycharmProjects\pythonProject1Yolo8work\ultralytics\img_5.png'
results = model(image_path)

# Access and print results
if hasattr(results, 'print'):
    results.print()

# Display the image with bounding boxes
if hasattr(results, 'show'):
    results.show()

# Access bounding boxes, confidences, and class IDs
if hasattr(results, 'boxes'):
    # Extract boxes, which contains all detections
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy  # Coordinates of the box
        conf = box.conf  # Confidence score
        cls_id = box.cls  # Class ID
        class_name = results.names[cls_id]  # Get class name using class ID

        print(f"Detected {class_name} with confidence {conf:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

# Save results
if hasattr(results, 'save'):
    results.save()  # Save processed images
