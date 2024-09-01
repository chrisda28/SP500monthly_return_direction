import copy
import yfinance as yf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

INDICES = {
    'SP500': '^GSPC',
    'Russell 2000': '^RUT',
    'Dow Jones': '^DJI',
    'Nasdaq': '^IXIC'
}


def get_daily_stock_data(ticker: str, time_period: str):
    """Get daily index data, format date column, and calc NaN value occurrence"""
    ticker = INDICES.get(ticker)
    if not ticker:
        raise ValueError(f"Invalid index name: {ticker}")
    stock = yf.Ticker(ticker)  # create stock object
    history = stock.history(period=time_period)  # specify data time period to receive
    df = pd.DataFrame(history)  # convert stock data to dataframe
    df = df.drop(['Dividends', 'Stock Splits'], axis=1)  # drop unneeded columns
    df = df.reset_index()  # making date column normal to
    # parse away unneeded time info that was included in the date values
    df["Date"] = pd.to_datetime(df['Date']).dt.date  # convert date column to datetime and proper format
    missing_count = df.isnull().sum()  # summing up all null values in df
    df.set_index("Date", inplace=True)  # setting date column as the index
    return df, missing_count


def prep_data(df):
    """split data for training, testing"""
    df['Monthly_Return'] = df['Close'].pct_change(periods=20)  # Approximately one month of trading days
    df['Target_Label'] = (df['Monthly_Return'].shift(-20) > 0).astype(int)  # Predict next month's direction, predict
    # 5 days into future create column for bool value, pos return = 1 neg return = 0,
    # the .shift moves up the daily return column by 20 to try to indicate next month return based on current return

    df['MA_20'] = df['Close'].rolling(window=20).mean()  # find moving averages
    df['MA_60'] = df['Close'].rolling(window=60).mean()
    df['Volatility'] = df['Monthly_Return'].rolling(window=60).std()

    df.dropna(inplace=True)

    total_rows = len(df)
    train_index = int(total_rows * 0.8)

    train_data = df.iloc[:train_index].copy()  # 80% training 20% testing
    test_data = df.iloc[train_index:].copy()

    x_train = train_data.drop(['Target_Label', 'Monthly_Return'], axis=1)  # Dropping 'Daily_Return' as it's too
    # closely related to the target label (next day's return direction) and 'Target_Label'
    # as it's what model trying to predict
    y_train = train_data['Target_Label']

    x_test = test_data.drop(['Target_Label', 'Monthly_Return'], axis=1)
    y_test = test_data['Target_Label']

    return x_train, y_train, x_test, y_test


def run_model(ticker, time_period):
    df, missing_count = get_daily_stock_data(ticker=ticker, time_period=time_period)
    x_train, y_train, x_test, y_test = prep_data(df=df)
    lr_classifier = LogisticRegression()
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    lr_classifier.fit(X=x_train_scaled, y=y_train)
    predictions = lr_classifier.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    return predictions, accuracy, y_test, lr_classifier


def plot_confusion_matrix(y_true, y_pred):
    mtx = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.75,  cbar=False, ax=ax, cmap='Blues', linecolor='white')
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    static_dir = os.path.join(os.getcwd(), 'static')
    img_filename = 'confusion_matrix.png'
    img_path = os.path.join(static_dir, img_filename)  # getting img pathway so it can be saved
    plt.savefig(img_path, format='png', dpi=300)  # saving plot as an image to be used in flask app
    plt.close(fig)  # closing matplotlib figure to free up memory

    print(f"Plot saved to {img_path}")
    return img_filename


def feature_importance(model, x_train):
    """Graphically display feature importance"""
    feature_importance = pd.DataFrame({'feature': x_train.columns, 'importance': abs(model.coef_[0])})
    return feature_importance


