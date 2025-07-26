import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
today = date.today()

# Get today's date as a string for the end date of data download
d1 = today.strftime("%Y-%m-%d")
end_date = d1

# Calculate the start date (5000 days ago) for historical data
d2 = date.today() - timedelta(days = 5000)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

# Download historical stock data for Apple (AAPL) from Yahoo Finance
data = yf.download("AAPL", start=start_date, end=end_date, progress=False, auto_adjust=False)

# Add the date as a column and select relevant columns for analysis
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)
print(data.tail())  # Display the last few rows of the data

# Plot a candlestick chart of the stock prices using Plotly
fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'])])
fig.update_layout(title='AAPL Stock Price',
                  xaxis_rangeslider_visible=False)
fig.show()

# Prepare features (Open, High, Low, Volume) and target (Close) for model training
x = data[["Open", "High", "Low", "Volume"]]
y = data["Close"]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)  # Reshape target to be a column vector

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (xtrain.shape[1], 1)))  # First LSTM layer
model.add(LSTM(64, return_sequences=False))  # Second LSTM layer
model.add(Dense(25))  # Dense layer with 25 units
model.add(Dense(1))   # Output layer for regression
model.summary()       # Print model summary

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size=1, epochs=30)

import numpy as np
# Prepare a sample feature set for prediction (Open, High, Low, Volume)
features = np.array([[177.089996, 180.419998, 177.070007, 74919600]])
model.predict(features)  # Predict the closing price for the given features