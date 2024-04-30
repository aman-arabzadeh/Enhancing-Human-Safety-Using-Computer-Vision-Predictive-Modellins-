import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV
#df = pd.read_csv('tracking_and_predictions_DR.csv') #Dead Reckoning
df = pd.read_csv('tracking_and_predictions.csv')# Kalman Filtering

# Filter data for a specific class, e.g., 'cell phone'
class_name = 'cell phone'
df_filtered = df[df['class_name'] == class_name]

# Count the number of detections and predictions
num_detections = df_filtered['det_x'].dropna().count()
num_predictions = df_filtered['pred_x'].dropna().count()

# Plotting
plt.figure(figsize=(12, 8))

# Plot detected positions with blue circles
plt.scatter(df_filtered['det_x'], df_filtered['det_y'], c='blue', label=f'Detected Positions - {num_detections}', zorder=2, alpha=0.6)

# Plot predicted future positions with red crosses
plt.scatter(df_filtered['pred_x'], df_filtered['pred_y'], c='red', marker='x', label=f'Predicted Positions - {num_predictions}', zorder=1, alpha=0.6)

# Title with counts
plt.title(f'Tracking and Prediction for "{class_name}" | Detected: {num_detections}, Predicted: {num_predictions}', pad=20)

# Labels and Legend
plt.xlabel('X Position', labelpad=10)
plt.ylabel('Y Position', labelpad=10)
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
plt.grid(True, linestyle='--', alpha=0.5)

# Invert both x and y axes to rotate the plot 180 degrees
#plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.tight_layout(pad=2)

# Display the plot
plt.show()
