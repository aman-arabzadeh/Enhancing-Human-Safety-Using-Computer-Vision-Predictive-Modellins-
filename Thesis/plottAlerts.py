import pandas as pd
import matplotlib.pyplot as plt


# Load the data from the CSV file
data = pd.read_csv('alert_times_DR.csv')



# Plot current and predicted locations of objects
plt.figure(figsize=(12, 6))
plt.scatter(data['Object Location X (px)'], data['Object Location Y (px)'], color='blue', label='Current Location', alpha=0.6)
plt.scatter(data['Predicted Future Location X (px)'], data['Predicted Future Location Y (px)'], color='red', label='Predicted Location', alpha=0.6)
plt.title('Current and Predicted Locations of Detected Objects')
plt.xlabel('Location X (px)')
plt.ylabel('Location Y (px)')
plt.legend()
plt.grid(True)
plt.show()

# Plot a histogram of hazard times
plt.figure(figsize=(12, 6))
plt.hist(data['Hazard Time Since Start (seconds)'], bins=30, color='green', alpha=0.7)
plt.title('Distribution of Hazard Times Since Start')
plt.xlabel('Hazard Time Since Start (seconds)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# For each Alert Type, plot the Hazard Time
plt.figure(figsize=(12, 6))
for alert_type in data['Alert Type'].unique():
    subset = data[data['Alert Type'] == alert_type]
    plt.hist(subset['Hazard Time Since Start (seconds)'], bins=30, alpha=0.5, label=f'{alert_type}')

plt.title('Hazard Time Distribution by Alert Type')
plt.xlabel('Hazard Time Since Start (seconds)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

