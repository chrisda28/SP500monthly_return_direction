from flask import Flask, render_template, redirect
from data import *

app = Flask(__name__, static_folder='static')   # create an instance of flask application

SP500 = "^GSPC"
TIME_PERIOD = "10y"  # specifies how much data to collect,
# must be one of ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

@app.route("/")
def home():
    """render the home page"""
    return render_template(template_name_or_list='index.html', title='Stock Daily Return Direction Predicting',
                           header='',)


@app.route("/model")
def model():
    df, missing_count = get_daily_stock_data(ticker=SP500, time_period=TIME_PERIOD)
    x_train, y_train, x_test, y_test = prep_data(df=df)
    predictions, accuracy, y_test = run_model(ticker=SP500, time_period=TIME_PERIOD)

    return render_template(template_name_or_list='model.html')



@app.route("/evaluate")
def evaluate():



    return render_template(template_name_or_list='evaluate.html')








if __name__ == "__main__":
    app.run(debug=True)