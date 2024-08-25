import copy
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


TIME_PERIOD = "10y"  # specifies how much data to collect,
# must be one of ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']


def get_daily_stock_data(ticker: str, time_period: TIME_PERIOD):
    """Get daily stock data, format date column, and calc NaN value occurrence"""
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

    x_train = train_data.drop(['Target_Label', 'Monthly_Return'], axis=1) # Dropping 'Daily_Return' as it's too
    # closely related to the target label (next day's return direction) and 'Target_Label'
    # as it's what model trying to predict
    y_train = train_data['Target_Label']

    x_test = test_data.drop(['Target_Label', 'Monthly_Return'], axis=1)
    y_test = test_data['Target_Label']

    return x_train, y_train, x_test, y_test


def plot_confusion_matrix(y_true, y_pred):
    mtx = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.75,  cbar=False, ax=ax, cmap='Blues', linecolor='white')
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    plt.show()


df, missing_count = get_daily_stock_data("^GSPC", TIME_PERIOD)
x_train, y_train, x_test, y_test = prep_data(df=df)
# print(y_train.value_counts(), y_test.value_counts())
# print(df, missing_count)

print("Shape of x_train:", x_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of x_test:", x_test.shape)
print("Shape of y_test:", y_test.shape)

lr_classifier = LogisticRegression()  # initialize model
scaler = StandardScaler()  # why am i scaling again? is it bc of CLASS IMBALANCE
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
print("Columns in x_train:", x_train.columns)
print("Shape of x_train_scaled:", x_train_scaled.shape)
print("Shape of x_test_scaled:", x_test_scaled.shape)

lr_classifier.fit(X=x_train_scaled, y=y_train)
predictions = lr_classifier.predict(x_test_scaled)



print("Accuracy:", accuracy_score(y_test, predictions))
print("Class distribution in y_train:", y_train.value_counts(normalize=True))
print("Class distribution in y_test:", y_test.value_counts(normalize=True))

feature_importance = pd.DataFrame({'feature': x_train.columns, 'importance': abs(lr_classifier.coef_[0])})
print(feature_importance.sort_values('importance', ascending=False))
print(predictions)
print('The Model Accuracy on The Validation Data Was:', round(accuracy_score(y_test, predictions), 3)*100, '%')
# boo = plot_confusion_matrix(y_true=y_test, y_pred=predictions)
# print(boo)
