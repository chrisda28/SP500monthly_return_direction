import yfinance as yf
import pandas as pd
from datetime import datetime


def get_daily_stock_data(ticker: str, time_period: str):
    """Get daily stock data, calculate daily return and add column to DF that is returned,
    also find count of missing values"""
    stock = yf.Ticker(ticker)  # create stock object
    history = stock.history(period=time_period)  # specify data time period to receive
    df = pd.DataFrame(history)  # convert stock data to dataframe

    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change()  # calc daily returns

    # Fill NaN value in the first row with 0
    df['Daily_Return'].fillna(0)
    df = df.drop(['Volume', 'Dividends', 'Stock Splits'], axis=1)  # drop unneeded columns
    df = df.dropna()  # dropping first row with NaN value due to daily return calc
    missing_count = df.isnull().sum()
    return df, missing_count


df, missing_count = get_daily_stock_data("NVDA", "5y")

print(df, missing_count)