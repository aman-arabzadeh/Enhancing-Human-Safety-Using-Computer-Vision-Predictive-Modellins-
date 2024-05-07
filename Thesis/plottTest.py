import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def plot_predictions(filename, title):
    # Load the data
    data = pd.read_csv(filename)

    # Shift the actual detected positions to align them for comparison with the next prediction
    actual_next = data[['det_x', 'det_y']].shift(-1)
    data['next_actual_x'] = actual_next['det_x']
    data['next_actual_y'] = actual_next['det_y']

    # Drop the last row because it will have NaN values for 'next_actual_x' and 'next_actual_y' after the shift
    data = data[:-1]

    # Calculate MAE and MSE between predicted points and the next actual points
    mae = mean_absolute_error(data[['pred_x', 'pred_y']].values, data[['next_actual_x', 'next_actual_y']].values)
    mse = mean_squared_error(data[['pred_x', 'pred_y']].values, data[['next_actual_x', 'next_actual_y']].values)

    # Plotting
    plt.figure(figsize=(10, 6))
    for index, row in data.iterrows():
        plt.plot([row['pred_x'], row['next_actual_x']], [row['pred_y'], row['next_actual_y']], 'r-', marker='o')
        plt.text(row['pred_x'], row['pred_y'], f'Pred {index}', fontsize=8, ha='right')
        plt.text(row['next_actual_x'], row['next_actual_y'], f'Actual {index+1}', fontsize=8, ha='left')

    plt.title(f"{title}\nMAE: {mae:.2f}, MSE: {mse:.2f}")
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True)
    plt.show()

    # Optionally return MAE and MSE if you need these values for further analysis
    return mae, mse

# Usage examples:
plot_predictions('tracking_and_predictions.csv', 'Predicted Position to Subsequent Actual Position - Kalman Filtering Parabolic Movements')
plot_predictions('tracking_and_predictions_DR.csv', 'Predicted Position to Subsequent Actual Position - Dead Reckoning Parabolic Movements')
plot_predictions('tracking_and_predictions_linear.csv', 'Predicted Position to Subsequent Actual Position - Kalman Filtering Linear Movements')
plot_predictions('tracking_and_predictions_DRLinear.csv', 'Predicted Position to Subsequent Actual Position - Dead Reckoning Linear Movements')
