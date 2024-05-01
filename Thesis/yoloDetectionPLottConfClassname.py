import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend('TkAgg')

# Load data from CSV
data = pd.read_csv('yolo_data.csv')

# Compute the average confidence scores for each class
average_confidences = data.groupby('Class Name')['Confidence Score'].mean()

# Printing average confidences
print("Average Confidence Scores by Class:")
print(average_confidences)

# Plotting
plt.figure(figsize=(8, 5))  # Set the figure size
ax = average_confidences.plot(kind='bar', color=['blue', 'green'])  # Create a bar chart with an axis variable
plt.title('Average Confidence Scores by Class for DR Dynamic movements')  # Title of the plot
plt.xlabel('Class Name')  # Label for the x-axis
plt.ylabel('Average Confidence Score')  # Label for the y-axis
plt.xticks(rotation=0)  # Rotate class names for better readability

# Adding full precision text labels on the bars
for i, v in enumerate(average_confidences):
    ax.text(i, v + 0.01, f"{v:.5f}", color='black', ha='center')  # Adjust the precision as needed

plt.show()  # Display the plot
