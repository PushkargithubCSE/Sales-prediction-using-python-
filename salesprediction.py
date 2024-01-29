import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('advertising.csv')

print(data.head())  # Display the first few rows of the DataFrame
print(data.info())  # Display summary information about the DataFrame
print(data.describe())  # Display summary statistics of the numerical columns

model = LinearRegression()

model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

new_data = [[100, 25, 10]]  # Example: $100k on TV, $25k on Radio, $10k on Newspaper
predicted_sales = model.predict(new_data)
print("Predicted Sales:", predicted_sales)
