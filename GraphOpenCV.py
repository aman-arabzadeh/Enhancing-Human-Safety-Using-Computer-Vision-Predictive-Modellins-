import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data from CSV
try:
    df = pd.read_csv('tracking_and_predictions.csv')
except Exception as e:
    print(f"An error occurred: {str(e)}")
    exit()

# Ensure proper data types
df[['det_x', 'det_y', 'pred_x', 'pred_y']] = df[['det_x', 'det_y', 'pred_x', 'pred_y']].astype(float)

# Filter data for a specific class, e.g., 'cell phone'
class_name = 'cell phone'
df_filtered = df[df['class_name'] == class_name]

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot detected positions over time
ax.scatter(df_filtered['timestamp'], df_filtered['det_x'], df_filtered['det_y'], c='blue', label='Detected Path', alpha=0.7, s=50)

# Plot predicted future positions over time
ax.scatter(df_filtered['timestamp'], df_filtered['pred_x'], df_filtered['pred_y'], c='red', marker='x', label='Predicted Future Path', alpha=0.7, s=50)

ax.set_title(f'3D Trajectory and Prediction for {class_name}')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('X Position')
ax.set_zlabel('Y Position')
ax.legend()

plt.show()
