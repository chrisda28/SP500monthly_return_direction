from flask import Flask, render_template, redirect
import pandas as pd
from data import *

app = Flask(__name__, static_folder='static')   # create an instance of flask application

SP500 = "^GSPC"
TIME_PERIOD = "10y"  # specifies how much data to collect,
# must be one of ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']


@app.route("/")
def home():
    """render the home page/introduction"""
    return render_template(template_name_or_list='index.html', title='SP500 Monthly Return Direction Predicting',
                           header='',)


@app.route("/model")
def model():
    """Run model and display the prediction for future months"""
    predictions, accuracy, y_test, lr_classifier = run_model(ticker=SP500, time_period=TIME_PERIOD)
    y_test = y_test.squeeze() if isinstance(y_test, pd.DataFrame) else y_test
    predictions = pd.Series(predictions)
    result = pd.concat([y_test, predictions], axis=1, keys=['Actual', 'Predicted'])
    result.reset_index(inplace=True)
    result.rename(columns={'index': 'Date'}, inplace=True)
    halfway_point = len(result) // 2
    result.loc[:halfway_point - 1, 'Predicted'] = result.loc[halfway_point:, 'Predicted'].values
    result = result.iloc[:halfway_point]
    result['Matches'] = result['Actual'] == result['Predicted']
    result = result[['Date', 'Actual', 'Predicted', 'Matches']]
    result = result.iloc[::-1]  # reverse the order of rows so the newest dates are on top in table
    results = result.to_dict('records')  # convert df to list of dictionaries
    return render_template(template_name_or_list='model.html', results=results)


@app.route("/evaluate")
def evaluate():
    """Show key model metrics"""
    df, missing_count = get_daily_stock_data(ticker=SP500, time_period=TIME_PERIOD)
    x_train, y_train, x_test, y_test = prep_data(df=df)
    predictions, accuracy, y_test, lr_classifier = run_model(ticker=SP500, time_period=TIME_PERIOD)
    confusion_img = plot_confusion_matrix(y_true=y_test, y_pred=predictions)
    importance = feature_importance(model=lr_classifier, x_train=x_train)
    importance_data = importance.to_dict('records')
    return render_template(template_name_or_list='evaluate.html',
                           confusion=confusion_img,
                           accuracy=round(accuracy * 100, 2),
                           importance=importance_data)


if __name__ == "__main__":
    app.run(debug=True)
