import yfinance as yf
import pandas as pd
from datetime import datetime


def get_daily_stock_data(ticker: str, time_period: str):
    stock = yf.Ticker(ticker)
    history = stock.history(period=time_period)
    df = pd.DataFrame(history)

    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change()

    # Fill NaN value in the first row with 0
    df['Daily_Return'].fillna(0)
    df = df.drop(['Volume', 'Dividends', 'Stock Splits'], axis=1)
    return df


df = get_daily_stock_data("NVDA", "5y")

print(df.columns)