import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
#data = pd.read_csv('alert_times_DR.csv')
data = pd.read_csv('alert_times.csv')
# Convert timestamps to readable datetime objects
data['Pre-alert DateTime UTC'] = pd.to_datetime(data['Pre-alert DateTime UTC'], unit='s')
data['Post-alert DateTime UTC'] = pd.to_datetime(data['Post-alert DateTime UTC'], unit='s')

# Plot Alert Durations by Object Type
plt.figure(figsize=(10, 6))
sns.barplot(x='Detected Object Type', y='Alert Duration (seconds)', data=data)
plt.title('Alert Duration by Object Type')
plt.xlabel('Object Type')
plt.ylabel('Duration (seconds)')
plt.xticks(rotation=45)
plt.show()

# Plot Object Movements with Central Area Highlighted
plt.figure(figsize=(12, 8))
for index, row in data.iterrows():
    plt.plot([row['Object Location X (px)'], row['Predicted Future Location X (px)']],
             [row['Object Location Y (px)'], row['Predicted Future Location Y (px)']],
             marker='o', markersize=5, label=f"Obj {index+1} ({row['Detected Object Type']})" if index < 10 else "")
    plt.text(row['Object Location X (px)'], row['Object Location Y (px)'], str(index+1))

# Assume the central area coordinates are consistent across all rows, use the first row to define the rectangle
top_left_x = data.iloc[0]['Center Area Top-Left X (px)']
top_left_y = data.iloc[0]['Center Area Top-Left Y (px)']
bottom_right_x = data.iloc[0]['Center Area Bottom-Right X (px)']
bottom_right_y = data.iloc[0]['Center Area Bottom-Right Y (px)']

# Debugging output of the central area coordinates
print("Central Area Coordinates:")
print(f"Top-Left: ({top_left_x}, {top_left_y})")
print(f"Bottom-Right: ({bottom_right_x}, {bottom_right_y})")

# Draw a rectangle for the central area with a label
rect = plt.Rectangle((top_left_x, top_left_y), bottom_right_x - top_left_x, bottom_right_y - top_left_y,
                                  fill=False, edgecolor='red', linewidth=2)
plt.gca().add_patch(rect)
plt.text(top_left_x, top_left_y - 10, 'Robotic Arm Area', color='red', fontsize=12, ha='left')

plt.title('Object Movement from Current to Predicted Future Location with THE AREA OF ROBOTIC ARM')
plt.xlabel('X Coordinate (px)')
plt.ylabel('Y Coordinate (px)')
plt.legend(title='Object ID')
# Invert both x and y axes to rotate the plot 180 degrees
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

# Plot Hazard Time Distribution by Alert Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='Alert Type', y='Hazard Time Since Start (seconds)', data=data)
plt.title('Hazard Time Distribution by Alert Type')
plt.xlabel('Alert Type')
plt.ylabel('Hazard Time Since Start (seconds)')
plt.show()
