from flask import Flask, render_template


app = Flask(__name__, static_folder='static')   # create an instance of flask application


@app.route("/")
def home():
    """render the home page"""
    return render_template(template_name_or_list='home.html', title='Stock Daily Return Direction Predicting',
                           header='',)