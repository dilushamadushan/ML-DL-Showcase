import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go
from autots import AutoTS

# Step 1: Get today's date and compute start date
today = date.today()
end_date = today.strftime("%Y-%m-%d")
start_date = (today - timedelta(days=768)).strftime("%Y-%m-%d")

# Step 2: Download BTC-USD price data
data = yf.download('BTC-USD',
                   start=start_date,
                   end=end_date,
                   progress=False,
                   auto_adjust=False)

# Step 3: Clean and prepare the DataFrame
if not data.empty:
    data["Date"] = data.index
    data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    data.reset_index(drop=True, inplace=True)

    # Step 4: Show sample data
    print("First 5 rows of the data:")
    print(data.head())
    print("\nLast 5 rows of the data:")
    print(data.tail())
    print(f"\nDataset shape: {data.shape}")

    # Step 5: Plot candlestick chart only if data is clean
    if data[["Open", "High", "Low", "Close"]].notna().all().all():
        figure = go.Figure(data=[
            go.Candlestick(x=data["Date"],
                           open=data["Open"],
                           high=data["High"],
                           low=data["Low"],
                           close=data["Close"])
        ])
        figure.update_layout(title="Bitcoin Price Analysis",
                             xaxis_rangeslider_visible=False)
        figure.show()
    else:
        print("\n[WARNING] NaN found in OHLC data. Skipping candlestick chart.")

    # Step 6: Calculate and print correlation with 'Close'
    correlation = data.corr(numeric_only=True)
    try:
        close_corr = correlation["Close"]
        print("\nCorrelation with 'Close' price:")
        print(close_corr.sort_values(ascending=False))
    except Exception as e:
        print("\n[ERROR] Correlation failed:", str(e))
        print(correlation)

    # Step 7: Prepare for AutoTS Forecasting
    forecast_data = data[["Date", "Close"]].copy()
    forecast_data.columns = ["Date", "Close"]

    # Step 8: AutoTS model
    model = AutoTS(
        forecast_length=30,
        frequency='infer',
        ensemble='simple',
        model_list='fast',
        transformer_list='fast',
        drop_most_recent=1
    )

    model = model.fit(
        forecast_data,
        date_col='Date',
        value_col='Close',
        id_col=None
    )

    # Step 9: Forecast and print results
    prediction = model.predict()
    forecast = prediction.forecast
    print("\nForecasted Close Prices for the next 30 days:")
    print(forecast)

else:
    print("[ERROR] No data returned from Yahoo Finance. Please check your internet or date range.")
