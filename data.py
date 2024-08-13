import yfinance as yf
import pandas as pd
from datetime import datetime


def get_daily_stock_data(ticker: str, time_period: str):
    stock = yf.Ticker(ticker)
    history = stock.history(period=time_period)
    df = pd.DataFrame(history)

    return df

# example usage   df = get_daily_stock_data("NVDA", "5y")