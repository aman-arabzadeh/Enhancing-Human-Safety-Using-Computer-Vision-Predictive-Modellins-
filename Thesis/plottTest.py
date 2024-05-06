import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# https://www.pexels.com/sv-se/foto/102104/

data = pd.read_csv("tracking_and_predictions.csv")

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

# Filter the data for a specific class
clsname = "apple"
class_data = data[data['class_name'] == clsname]

# Separate actual and predicted coordinates
actual = class_data[['det_x', 'det_y']].values
predicted = class_data[['pred_x', 'pred_y']].values

# Calculate MAE and MSE
def mean_absolute_error(actual, predicted):
    """Calculate the Mean Absolute Error using Euclidean distance between actual and predicted points."""
    actual = np.array(actual)
    predicted = np.array(predicted)
    distances = np.sqrt(np.sum((actual - predicted) ** 2, axis=1))
    return np.mean(distances)

def mean_squared_error(actual, predicted):
    """Calculate the Mean Squared Error using the squared Euclidean distance between actual and predicted points."""

    actual = np.array(actual)
    predicted = np.array(predicted)
    squared_distances = np.sum((actual - predicted) ** 2, axis=1)
    return np.mean(squared_distances)

mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

# Create a figure with larger size
plt.figure(figsize=(15, 10))

# Plot actual and predicted x-coordinates
plt.plot(class_data['timestamp'], class_data['det_x'], label=f'{clsname} det_x')
plt.plot(class_data['timestamp'], class_data['pred_x'], label=f'{clsname} pred_x')

# Add general plot formatting
plt.legend()
plt.title('Tracking and Prediction Comparison')
plt.xlabel('Timestamp')
plt.ylabel('Coordinates')
plt.tight_layout()
plt.show()
