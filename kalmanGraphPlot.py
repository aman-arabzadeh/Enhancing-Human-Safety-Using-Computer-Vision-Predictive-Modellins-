import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV with error handling
try:
    df = pd.read_csv('tracking_and_predictions.csv')
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()
except pd.errors.EmptyDataError:
    print("File is empty.")
    exit()
except pd.errors.ParserError:
    print("Error while parsing the file. Please check the file format.")
    exit()

# Ensure proper data types
df[['det_x', 'det_y', 'pred_x', 'pred_y']] = df[['det_x', 'det_y', 'pred_x', 'pred_y']].astype(float)

# Filter data for a specific class, e.g., 'person'
class_name = 'cup'
df_filtered = df[df['class_name'] == class_name]

# Plotting
plt.figure(figsize=(12, 8))

# Plot detected positions with blue circles
plt.scatter(df_filtered['det_x'], -df_filtered['det_y'], c='blue', label=f'Detected Path ({class_name})', zorder=2, alpha=0.7, s=50)

# Plot predicted future positions with red crosses
plt.scatter(df_filtered['pred_x'], -df_filtered['pred_y'], c='red', marker='x', label=f'Predicted Future Position ({class_name})', zorder=1, alpha=0.7, s=50)

plt.title(f'Object Tracking and Future Position Prediction for {class_name}')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move the legend outside of the plot
plt.grid(True, linestyle='--', alpha=0.5)  # Add gridlines
plt.tight_layout()  # Adjust the padding between and around subplots

plt.show()
