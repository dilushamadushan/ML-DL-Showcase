# Cryptocurrency Price Prediction using AutoTS and Yahoo Finance

This project downloads historical Bitcoin (BTC-USD) price data from Yahoo Finance, performs exploratory analysis including visualization and correlation, and then forecasts future prices using the AutoTS time series forecasting library.

---

## Features

- Download historical Bitcoin price data (Open, High, Low, Close, Volume) for the last ~2 years  
- Display candlestick chart for visualizing price movements  
- Compute correlation of different price features with the closing price  
- Forecast next 30 days Bitcoin closing prices using AutoTS ensemble models  

---

## Installation

Make sure you have Python 3.7+ installed. Then install the required packages:

```bash
pip install yfinance autots
```