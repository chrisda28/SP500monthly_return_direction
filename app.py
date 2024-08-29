from flask import Flask, render_template, redirect
from data import *

app = Flask(__name__, static_folder='static')   # create an instance of flask application

SP500 = "^GSPC"
TIME_PERIOD = "10y"  # specifies how much data to collect,
# must be one of ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']


@app.route("/")
def home():
    """render the home page/introduction"""
    return render_template(template_name_or_list='index.html', title='Stock Daily Return Direction Predicting',
                           header='',)


@app.route("/model")
def model():
    """Run model and display the prediction for future months"""
    df, missing_count = get_daily_stock_data(ticker=SP500, time_period=TIME_PERIOD)  # DONT THINK I NEED THIS LINE OR THE ONE BELOW
    x_train, y_train, x_test, y_test = prep_data(df=df)
    predictions, accuracy, y_test = run_model(ticker=SP500, time_period=TIME_PERIOD)
    # map the predictions to readable output instead of 1s and 0s !!!!!!!
    return render_template(template_name_or_list='model.html', predictions=predictions)


@app.route("/evaluate")
def evaluate():
    """Show key model metrics"""
    df, missing_count = get_daily_stock_data(ticker=SP500, time_period=TIME_PERIOD)
    x_train, y_train, x_test, y_test = prep_data(df=df)
    predictions, accuracy, y_test, lr_classifier = run_model(ticker=SP500, time_period=TIME_PERIOD)
    accuracy = round(accuracy, 4)
    confusion_img = plot_confusion_matrix(y_true=y_test, y_pred=predictions)
    importance = feature_importance(model=lr_classifier, x_train=x_train)
    importance_data = importance.to_dict('records')
    return render_template(template_name_or_list='evaluate.html',
                           confusion=confusion_img,
                           accuracy=accuracy,
                           importance=importance_data)








if __name__ == "__main__":
    app.run(debug=True)