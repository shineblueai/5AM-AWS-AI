import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 


# Mean Absolute Error (MAE)
# Error means the difference between the actual value and the predicted value.

def mean_absolute_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE) between the true and predicted values.

    Args:
        y_true (array-like): The true values.
        y_pred (array-like): The predicted values.

    Returns:
        float: The Mean Absolute Error.
    """
    return np.mean(np.abs(y_true - y_pred))

# Example data 
y_true = np.array([3 , -0.5, 2, 7 ])
y_pred = np.array([2.5, 0.0, 2, 8 ])

mae = mean_absolute_error(y_true, y_pred)
print("Mean Absolute Error:", mae)

#plotting the graph
plt.figure(figsize=(10, 10))
plt.scatter(y_true, y_pred, color='blue', label='Actual vs Predicted')
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()

