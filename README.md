# Index Monthly Return Classifier
#### Video Demo:  <URL HERE>
#### Description:
This is a logistic regression model that makes prediction about the next month's monthly returns on an
index being positive or negative (1 or 0). The model is showcases within a flask web application where the testing data and predictions are
shown. The user can specify between four different indexes (S&P 500, Russell 2000, DJIA,
 or NASDAQ). Then the user can specify different time frames for which the model will use, ranging from a single month
to over ten years. Also, the web application contains model metrics such as a confusion matrix, an accuracy score, and a table 
displaying model feature importance. 


## Navigating the Web Application

1. Home Page:

- Select your target index
- Choose analysis timeframe
- Click "Run Model" to generate predictions, and the different model metrics


2. Model Page:

- Displays a detailed predictions table, showing Test data actual values and predicted values
- User can compare actual vs. predicted values
- Analyze prediction accuracy


3. Key Metrics Page:

- User can see confusion matrix visualization, showing false positive predictions, false negative predictions, etc.
- Review model accuracy scores
- Examine feature importance rankings, to see which features contribute the most
## Technical Details of the Model
Model Features - Index data pulled (yfinance) and organized/cleaned with Pandas library

- Closing price
- Trading volume
- Price range (High - Low)
- 20-day moving average (short term trend)
- 60-day moving average (medium term trend)
- 60-day moving volatility

## Model Specifications

- Algorithm: Logistic Regression (scikit-learn implementation)
- Data Split: 80% training / 20% testing
- Feature Processing: StandardScaler normalization
- Target Variable: Binary classification

- 1: Positive monthly return
- 0: Negative monthly return

Time Window: 20-day trading period for monthly return calculations
## Data Sources and Processing

- Real time data fetching using Yahoo Finance API (yfinance)
-  Data cleaning and preparation using Pandas
- User Configurable time period selection (1 month to 10+ years)

## Navigating each File:
- data.py: gets a hold of the index data from Yahoo Finance. This data is formatted/cleaned/organized with Pandas and
then passed to the model. data.py contains the functionality that runs the model itself, as well as the functionality
that creates the confusion matrix as well as the feature importance table that is displayed within the web application.
- app.py: This file is responsible for setting up different flask routes. It decides which HTML templates are rendered
and passed user input into the functionality built within data.py
- layout.html: this is the baseline structure of the web application, each other .html file extends this one
- index.html: this is the homepage, it is capable of handling user input such as when they specify the time frame and 
index
- model.html: this displays the models test data. Showing the predicted values as well as the actual values used. It 
shows all of this in tabular format.
- evaluate.html: this displays the confusion matrix, the model's accuracy score, and the table containing the model's
feature importance. 
- style.css: This contains the styling for the website.

### The model supports four major market indices:

- S&P 500 (^GSPC)
- Russell 2000 (^RUT)
- Dow Jones Industrial Average (^DJI)
- NASDAQ Composite (^IXIC)


## AI Usage
AI was used to help create visualizations such as the confusion matrix. AI was also heavily used in the CSS
styling of the web application, except for the use of Bootstrap. AI also advised me on how to set up the model, as
this was my first time using a logistic regression without scikit-learn. I did not have any formal ML knowledge going
into this project, so AI helped answer general questions about how things work, and it helped me learn while I
implemented the model. I also used AI to help me create a plan for this project before any coding at all. I think
this really helped and I really benefited from thinking about the project before actually coding.

## Possible Future Improvements
1. Implement more advanced machine learning mechanisms to better understand index movements
2. Expand the analysis to include more indices
3. Implement user authentication to allow personalized watchlists on different indices
4. Add more interactive features to the visualizations, such as specific date range selection
5. Introduce nonlinearities in the model's features

## Author
Christian Asimou
