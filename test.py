import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('tracking_and_predictions.csv', header=None,
                   names=['timestamp', 'det_x', 'det_y', 'pred_x', 'pred_y', 'class_name'])

# Plotting
plt.figure(figsize=(10, 8))

# Plot detected positions
plt.scatter(data['det_x'], data['det_y'], alpha=0.6, color='blue', label='Detected Position')

# Plot predicted positions
plt.scatter(data['pred_x'], data['pred_y'], alpha=0.6, color='red', label='Predicted Position')

plt.title('Object Tracking Positions')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)

plt.show()


# Enhancing Human Safety Using Computer Vision and Predictive Modellings