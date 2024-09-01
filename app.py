from flask import Flask, render_template, redirect, request, session
import pandas as pd
from data import *

app = Flask(__name__, static_folder='static')   # create an instance of flask application
app.secret_key = 'secret'
SP500 = "^GSPC"
TIME_PERIOD = "10y"  # specifies how much data to collect,
# must be one of ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']


@app.route("/")
def home():
    """render the home page/introduction"""
    if request.method == 'POST':
        index_name = request.form.get('index_name')
        session['selected_index'] = index_name
    else:
        index_name = session.get('selected_index', 'S&P 500')  # Default to S&P 500

    return render_template(template_name_or_list='index.html',
                           title='Index Monthly Return Direction Predicting',
                           header='',
                           selected_index=index_name)


@app.route("/model", methods=['GET', 'POST'])
def model():
    """Run model and display the predictions corresponding to y_test data"""
    if request.method == 'POST':
        index_name = request.form.get('index_name')
        session['selected_index'] = index_name
    else:
        index_name = session.get('selected_index', 'S&P 500')  # Default to S&P 500

    predictions, accuracy, y_test, lr_classifier = run_model(ticker=index_name, time_period=TIME_PERIOD)
    # predictions, accuracy, y_test, lr_classifier = run_model(ticker=SP500, time_period=TIME_PERIOD)
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
    return render_template(template_name_or_list='model.html', results=results, selected_index=index_name)


@app.route("/evaluate", methods=['GET', 'POST'])
def evaluate():
    """Show key model metrics"""
    if request.method == 'POST':
        index_name = request.form.get('index_name')
        session['selected_index'] = index_name
    else:
        index_name = session.get('selected_index', 'S&P 500')

    df, missing_count = get_daily_stock_data(ticker=index_name, time_period=TIME_PERIOD)
    # df, missing_count = get_daily_stock_data(ticker=SP500, time_period=TIME_PERIOD)
    x_train, y_train, x_test, y_test = prep_data(df=df)
    predictions, accuracy, y_test, lr_classifier = run_model(ticker=index_name, time_period=TIME_PERIOD)
    confusion_img = plot_confusion_matrix(y_true=y_test, y_pred=predictions)
    importance = feature_importance(model=lr_classifier, x_train=x_train)
    importance_data = importance.to_dict('records')
    return render_template(template_name_or_list='evaluate.html',
                           confusion=confusion_img,
                           accuracy=round(accuracy * 100, 2),
                           importance=importance_data, selected_index=index_name)


if __name__ == "__main__":
    app.run(debug=True)
