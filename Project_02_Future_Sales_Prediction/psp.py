# 📦 Import necessary libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 📥 Load the dataset
print("Loading dataset...")
data = pd.read_csv("advertising.csv")
print("\nFirst 5 rows of the dataset:")
print(data.head())
print("\nChecking for missing values:")
print(data.isnull().sum())  # Check for null values

# 📊 Visualize relationships between Sales and each feature with trendline
print("\nDisplaying scatter plots with trendlines...")
figure = px.scatter(data_frame=data, x="Sales", y="TV", size="TV", trendline="ols",
                    title="Sales vs TV Advertising")
figure.show()

figure = px.scatter(data_frame=data, x="Sales", y="Newspaper", size="Newspaper", trendline="ols",
                    title="Sales vs Newspaper Advertising")
figure.show()

figure = px.scatter(data_frame=data, x="Sales", y="Radio", size="Radio", trendline="ols",
                    title="Sales vs Radio Advertising")
figure.show()

# 📈 Show correlation with Sales
print("\nCorrelation of features with Sales:")
correlation = data.corr()
print(correlation["Sales"].sort_values(ascending=False))

# 🧪 Prepare data for training
print("\nPreparing data for model training...")
X = np.array(data.drop(["Sales"], axis=1))  # Features: TV, Radio, Newspaper
y = np.array(data["Sales"])  # Target: Sales

# Split into training and testing sets (80% train, 20% test)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# 🤖 Train a Linear Regression model
print("\nTraining the Linear Regression model...")
model = LinearRegression()
model.fit(xtrain, ytrain)
print("Model training complete.")

# 📏 Evaluate model accuracy
accuracy = model.score(xtest, ytest)
print(f"\nModel accuracy on testing set: {accuracy:.2f}")

# 🔮 Make a prediction with new values
print("\nMaking prediction for new advertising budget:")
sample_features = np.array([[230.1, 37.8, 69.2]])  # [TV, Radio, Newspaper]
predicted_sales = model.predict(sample_features)
print(f"Predicted Sales for TV=230.1, Radio=37.8, Newspaper=69.2: {predicted_sales[0]:.2f}")
