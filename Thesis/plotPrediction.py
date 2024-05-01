import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV
#df = pd.read_csv('tracking_and_predictions.csv')  # Kalman Filtering
df = pd.read_csv('tracking_and_predictions.csv')  # Kalman Filtering

# Filter data for a specific class, e.g., 'cell phone'
class_name = 'sports ball'
df_filtered = df[df['class_name'] == class_name]

# Count the number of detections and predictions
num_detections = df_filtered['det_x'].dropna().count()
num_predictions = df_filtered['pred_x'].dropna().count()

# Determine start and end positions if sorted by sequence or time
start_position = df_filtered.iloc[0][['det_x', 'det_y']] if not df_filtered.empty else "Data not available"
end_position = df_filtered.iloc[-1][['det_x', 'det_y']] if not df_filtered.empty else "Data not available"

# Plotting
plt.figure(figsize=(12, 8))

# Plot detected positions with blue circles
plt.scatter(df_filtered['det_x'], df_filtered['det_y'], c='blue', label=f'Detected Positions - {num_detections}', zorder=2, alpha=0.6)

# Plot predicted future positions with red crosses
plt.scatter(df_filtered['pred_x'], df_filtered['pred_y'], c='red', marker='x', label=f'Predicted Positions - {num_predictions}', zorder=1, alpha=0.6)

# Title with counts and position information
plt.title(f'Tracking and Prediction for "{class_name}" | Detected: {num_detections}, Predicted: {num_predictions}\nStart: {start_position.values}, End: {end_position.values}, KF', pad=20)

# Labels and Legend
plt.xlabel('X Position', labelpad=10)
plt.ylabel('Y Position', labelpad=10)
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
plt.grid(True, linestyle='--', alpha=0.5)

# Invert y-axis to rotate the plot 180 degrees
plt.gca().invert_yaxis()

plt.tight_layout(pad=2)

# Display the plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Assuming 'data' DataFrame already has 'det_x', 'det_y', 'pred_x', 'pred_y'
errors = np.sqrt((df['det_x'] - df['pred_x'])**2 + (df['det_y'] - df['pred_y'])**2)
plt.hist(errors, bins=50, alpha=0.75)
plt.title('Histogram of Prediction Errors')
plt.xlabel('Error Distance')
plt.ylabel('Frequency')
plt.show()

# Display basic statistics
print("Mean Error:", np.mean(errors))
print("Median Error:", np.median(errors))
print("Standard Deviation of Errors:", np.std(errors))
