import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the file, skip the first row
data = pd.read_csv("tracking_and_predictions.csv", skiprows=1, header=None, names=["timestamp", "det_x", "det_y", "pred_x", "pred_y", "class_name"])

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

# Get unique class names
class_names = data['class_name'].unique()

# Create a figure with larger size
plt.figure(figsize=(15, 10))

# Loop through each class and plot its data
#for i, class_name in enumerate(class_names):
# Filter data for the current class
clsname = "frisbee"
class_data = data[data['class_name'] == clsname]


# Plot det_x vs timestamp
plt.plot(class_data['timestamp'], class_data['det_x'], label='det_x')

# Plot det_y vs timestamp
plt.plot(class_data['timestamp'], class_data['det_y'], label='det_y')

# Plot pred_x vs timestamp
plt.plot(class_data['timestamp'], class_data['pred_x'], label='pred_x')

# Plot pred_y vs timestamp
plt.plot(class_data['timestamp'], class_data['pred_y'], label='pred_y')

# Add legend
plt.legend()

# Add title and labels
plt.title(f'Object Detection Data for {clsname}')
plt.xlabel('Timestamp')
plt.ylabel('Coordinates')

# Manage layout
plt.tight_layout()

# Show plot
plt.show()
