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


TIME_PERIOD = "5y"  # specifies how much data to collect


def get_daily_stock_data(ticker: str, time_period: TIME_PERIOD):
    """Get daily stock data, format date column, and calc NaN value occurrence"""
    stock = yf.Ticker(ticker)  # create stock object
    history = stock.history(period=time_period)  # specify data time period to receive
    df = pd.DataFrame(history)  # convert stock data to dataframe

    # df = df.drop(['Volume', 'Dividends', 'Stock Splits'], axis=1)  # drop unneeded columns
    df = df.reset_index()  # making date column normal to
    # parse away unneeded time info that was included in the date values
    df["Date"] = pd.to_datetime(df['Date']).dt.date  # convert date column to datetime and proper format

    missing_count = df.isnull().sum()  # summing up all null values in df
    df.set_index("Date", inplace=True)  # setting date column as the index
    return df, missing_count


def prep_data(df):
    """split data for training, testing"""
    total_rows = len(df)
    train_index = int(total_rows*.8)  # eighty percent training, twenty percent testing

    train_data = df.iloc[:train_index].copy()  # gather first (oldest) 80% of rows
    test_data = df.iloc[train_index:].copy()  # gather top (newest) 20% of rows

    for dataset in [train_data, test_data]:
        dataset.loc[:, 'Daily_Return'] = dataset['Close'].pct_change()  # calc daily return (% change), make new column
        dataset.loc[:, 'Target_Label'] = (dataset['Daily_Return'].shift(-1) > 0).astype(int)
        # create column for bool value, pos return = 1 neg return = 0,  the .shift moves up the daily return
        # column by 1 to try to indicate next day return based on current return
        dataset.dropna(inplace=True)

    x_train = train_data.drop(['Target_Label', 'Daily_Return'], axis=1)  # Dropping 'Daily_Return' as it's too
    # closely related to the target label (next day's return direction) and 'Target_Label'
    # as it's what model trying to predict

    y_train = train_data['Target_Label']

    x_test = test_data.drop(['Target_Label', 'Daily_Return'], axis=1)
    y_test = test_data['Target_Label']

    return x_train, y_train, x_test,  y_test


def plot_confusion_matrix(y_true, y_pred):
    mtx = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.75,  cbar=False, ax=ax,cmap='Blues',linecolor='white')
    plt.ylabel('true label')
    plt.xlabel('predicted label')


df, missing_count = get_daily_stock_data("NVDA", "5y")
x_train, y_train, x_test, y_test = prep_data(df=df)
# print(y_train.value_counts())
# print(df, missing_count)

lr_classifier = LogisticRegression()  # initialize model

lr_classifier.fit(X=x_train, y=y_train)
predictions = lr_classifier.predict(x_test)
print(predictions)
print('The Model Accuracy on The Validation Data Was:', round(accuracy_score(y_test, predictions), 3)*100, '%')
