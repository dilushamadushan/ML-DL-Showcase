# Stock Price Prediction using LSTM (AAPL)

This project uses Long Short-Term Memory (LSTM) neural networks to predict Apple Inc. (AAPL) stock closing prices based on historical data.

The model leverages Yahoo Finance data and TensorFlow's Keras API for deep learning.

## Features

- Downloads historical stock data for AAPL using `yfinance`
- Visualizes stock prices with a candlestick chart (using `plotly`)
- Prepares data and trains an LSTM model to predict closing prices
- Makes predictions on sample data

## Installation

Make sure you have Python 3.7+ installed.

Install the required dependencies:

```bash
pip install yfinance plotly pandas numpy scikit-learn tensorflow
