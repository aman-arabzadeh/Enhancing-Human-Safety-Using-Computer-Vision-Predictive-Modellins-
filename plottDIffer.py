import pandas as pd
import matplotlib.pyplot as plt


# Load the data from the CSV file
data = pd.read_csv('alert_times.csv')
# Histogram of Response Times
plt.figure(figsize=(10, 5))
plt.hist(data['Response Time'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Response Times')
plt.xlabel('Response Time (seconds)')
plt.ylabel('Frequency')
plt.show()

# Bar Plot of Average Response Time by Object Class
average_response_by_object = data.groupby('Object Class')['Response Time'].mean().sort_values()
plt.figure(figsize=(10, 5))
average_response_by_object.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Average Response Time by Object Class')
plt.xlabel('Object Class')
plt.ylabel('Average Response Time (seconds)')
plt.show()

# Line Plot of Hazard Time vs. Alert Time
plt.figure(figsize=(12, 6))
plt.plot(data['Hazard Time'], label='Hazard Time', marker='o', linestyle='-', color='orange')
plt.plot(data['Alert Time'], label='Alert Time', marker='x', linestyle='--', color='blue')
plt.title('Hazard Time and Alert Time Over Records')
plt.xlabel('Record Number')
plt.ylabel('Time (seconds)')
plt.legend()
plt.show()