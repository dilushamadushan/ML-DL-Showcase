import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ------------------- Load Data -------------------
print("📂 Loading dataset...")
data = pd.read_csv('tips.csv')
print("\n🔍 First 5 records:")
print(data.head())

# ------------------- Data Visualization -------------------

print("\n📊 Visualizing data...")

fig1 = px.scatter(data, x='total_bill', y='tip', color='day', size='size', trendline='ols',
                  title='Tips vs Total Bill by Day')
fig1.show()

fig2 = px.scatter(data, x='total_bill', y='tip', color='sex', size='size', trendline='ols',
                  title='Tips vs Total Bill by Gender')
fig2.show()

fig3 = px.scatter(data, x='total_bill', y='tip', color='time', size='size', trendline='ols',
                  title='Tips vs Total Bill by Time')
fig3.show()

fig4 = px.pie(data, names='day', values='tip', title='Total Tips by Day', hole=0.5)
fig4.show()

fig5 = px.pie(data, names='time', values='tip', title='Total Tips by Time', hole=0.5)
fig5.show()

fig6 = px.pie(data, names='smoker', values='tip', title='Total Tips by Smoker Status', hole=0.5)
fig6.show()

# ------------------- Data Preprocessing -------------------

print("\n🧹 Preprocessing data...")

data["sex"] = data["sex"].map({"Female": 0, "Male": 1})
data["smoker"] = data["smoker"].map({"No": 0, "Yes": 1})
data["day"] = data["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
data["time"] = data["time"].map({"Lunch": 0, "Dinner": 1})

print("\n🔁 Encoded data:")
print(data.head())

# ------------------- Feature Selection -------------------

x = np.array(data[['total_bill', 'size', 'sex', 'smoker', 'day', 'time']])
y = np.array(data['tip'])

# ------------------- Train/Test Split -------------------

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# ------------------- Model Training -------------------

print("\n🤖 Training Linear Regression model...")
model = LinearRegression()
model.fit(x_train, y_train)

# ------------------- Prediction and Evaluation -------------------

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📉 Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# ------------------- Custom Prediction -------------------

print("\n🔮 Predicting tip for new data:")

# Sample: [total_bill, size, sex, smoker, day, time]
# Example: $24.50 bill, 1 person, Female(0), Non-smoker(0), Friday(1), Dinner(1)
feature = np.array([[24.50, 1, 0, 0, 1, 1]])

predicted_tip = model.predict(feature)
print(f"Predicted tip: ${predicted_tip[0]:.2f}")
