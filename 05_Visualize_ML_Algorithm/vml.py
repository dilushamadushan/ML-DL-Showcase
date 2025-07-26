import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the advertising dataset from a CSV file
print("Loading dataset from Advertising.csv...")
data = pd.read_csv("Advertising.csv")
print("First 5 rows of the dataset:")
print(data.head())  # Display the first few rows of the dataset

# Extract the 'TV' column as the feature and 'Sales' as the target variable
print("\nPreparing features and target variable...")
X = data['TV'].values.reshape(-1, 1)
y = data['Sales']

# Create and train a linear regression model using the TV advertising data
print("\nTraining Linear Regression model...")
model = LinearRegression()
model.fit(X, y)
print("Model training complete.")

# Generate a range of TV advertising values for plotting the regression line
x_range = np.linspace(X.min(), X.max(), 100)
y_range = model.predict(x_range.reshape(-1, 1))

import plotly.express as px
import plotly.graph_objects as go

# Create a scatter plot of the original data
print("\nGenerating scatter plot and regression line...")
fig = px.scatter(data, x='TV', y='Sales', title='TV Advertising vs Sales', opacity=0.65)

# Add the regression line to the plot
fig.add_trace(go.Scatter(x=x_range, y=y_range, name='Regression Line'))

# Show the plot
fig.show()
print("Plot displayed.")

# Print the predicted sales values for the generated TV advertising range
print("\nPredicted sales for TV advertising range:")