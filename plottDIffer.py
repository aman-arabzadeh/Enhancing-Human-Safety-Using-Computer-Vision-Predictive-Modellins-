import pandas as pd
import matplotlib.pyplot as plt


# Load the data from the CSV file
#data = pd.read_csv('alert_times.csv')
data = pd.read_csv('alert_times_DR.csv')
# Histogram of Response Times
plt.figure(figsize=(10, 5))
plt.hist(data['Response Time'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Response Times')
plt.xlabel('Response Time (seconds)')
plt.ylabel('Frequency')
plt.show()
"""

Hazard Time:
This is the timestamp at which a potential hazard is first detected by the system. 


Alert Time:
This is the timestamp when an alert is actually issued by the system in response to the detected hazard. 

Person Class:
This refers to the classification of the detected entity as a "person" in the object detection system.

Object Class:
This describes the type of object that has been detected as posing a potential hazard when in close proximity to a person. 

Response Time:
This is calculated as the difference between the "Alert Time" and the "Hazard Time". 
It measures how quickly the system responds to a detected hazard by issuing an alert. 
A shorter response time indicates a more effective and potentially safer system,
 as it reduces the window of risk for accidents or injuries.

"""
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