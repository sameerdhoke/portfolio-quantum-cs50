# Portfolio Quantum
### Video Demo: https://www.youtube.com/watch?v=48D8FFr1CUg

#### Name: Sameer Dhoke
#### Location: Mumbai, Maharashtra, India
#### Submission Date: 30-January-2023

### Description:

#### Introduction
This project aims to create a web application that analyzes portfolios using quantitative and machine learning techniques. The application consists of 4 pages:
* Index page
* ML Model page
* Momentum Strategy page
* Value Strategy page

#### Project Structure
The project directory is named project, and it contains 2 folders:

1. app.py - A Flask file that contains the server-side python code for the ML model
2. Three CSV files - sp_500_stocks.csv, value_metrics_sp500.csv, and momentum_metrics_sp500.csv
3. The templates folder contains the HTML templates for each of the pages mentioned above:
    * index.html for index page that also hase java script code
    * ml.html for ML model
    * value.html for Value model
    * momentum.html for Momentum model.
4. The static folder contains the following files:
    * styles.css - Deals with the styling of the webpage.
    * images folder - Contains 2 image files in PNG format.

#### Models
The following three models are implemented in the project:

1. ML Model: A brief explanation of the ML model and its implementation.
    * Brief explanation of ML model: <br>
    This is a machine-learning model to track the performance of a stock portfolio. Following is the outline of the ML model:
        * Collect historical data for the stocks in your portfolio. Here, a library called pandas-datareader is used to retrieve stock data from sources like Yahoo Finance.
        * Clean and preprocess the data. This includes handling missing values, converting dates to a suitable format, and normalizing the data.
        * Split the data into training (80% of the stock data) and testing sets (20% of the stock data). This is done to evaluate the model's performance on unseen data, so you should allocate a portion of your data for testing.
        * The machine learning model selected for this task is a linear regression model. We then train it on the training data.
        * Evaluate the model on the testing data. This will give you an idea of how well the model can predict the performance of the stocks in your portfolio.
        * Quantify the accuracy of the model using various evaluation metrics, such as mean absolute error (MAE), mean squared error (MSE), and root mean squared error (RMSE). These metrics can be calculated using functions from sklearn.metrics module.
        * Lower values for these metrics indicate a more accurate model. To assess the model's accuracy, you can also use other evaluation metrics, such as the R-squared value. The R-squared value measures how well the model fits the data, with a value between 0 and 1. A value of 1 indicates a perfect fit, while a value of 0 indicates that the model does not explain any variances in the data.
        * Use the trained model to make predictions on new, unseen data. These predictions are then used to make informed investment decisions.
        * To interpret the output of the evaluation metrics, you can compare the values and some baseline values.

    * Flask implementaion of ML model in flask app: <br>
        * The flask implementation of this model defines two flask routes, '/mlmodel' and '/mlmodel', which are defined using the @app.route decorator. The first route, /mlmodel, handles GET requests and returns a form for entering stock information. It reads a CSV file named sp_500_stocks.csv into a Pandas DataFrame named stocks, converts the compName column of the stocks DataFrame into a list, and assigns it to the options variable. An HTML string containing a datalist element is generated, with option elements for each item in the options list. This HTML string is then passed to the render_template function, which is used to render a template file named 'ml.html', with the HTML argument being passed to the template. <br>
        * The second route, /mlmodel, handles POST requests and processes the information entered in the form. It retrieves the form data entered by the user, including the stock name, stock quantity, and purchase price per share. If the form data includes a value named 'done', the application checks if the number of stocks entered is less than 5. If it is, the template is re-rendered with an error message indicating that at least 5 stocks are required. If the number of stocks is greater than or equal to 5, the application creates an empty Pandas DataFrame ml_dataframe with 3 columns and populates it with the stock information from the form. It then matches the Name of Stocks in the ml_dataframe with the compName in the stocks DataFrame using the process.extractOne method from the fuzzywuzzy library. If a match is found, the corresponding Ticker value is extracted and added to the ml_dataframe. <br>
        * The application then creates an empty Pandas DataFrame named stock_data, retrieves a list of all the ticker symbols from the ml_dataframe, and retrieves the adjusted close price data for each ticker symbol using the yf.download function from the yfinance library. The stock data is then used to perform additional processing and generate a plot, which is saved as an image file and returned to the user.

    * HTML template of ML model: <br>
    The HTML template "ml.html" uses bootstrap, a CSS framework, for styling, and has links to several pages, including a home page, a machine learning model to analyze a portfolio, and two quantitative investment strategies (momentum and value).
        * The head of the document includes links to external CSS and JavaScript files that provide the styling and functionality of the page. It also contains the title of the web page. The document's body contains the web page's main content, including the navigation bar, form, and other elements. The navigation bar is created using Bootstrap CSS and JavaScript. It provides links to different sections of the web page.

        * The form allows the user to enter information about the stocks in their portfolio. It then uses the POST method to submit the data to the server. The drop-down menu is created using HTML5's datalist element. It provides a list of options for the user. The error message is displayed when there is a problem with the user's input. It is generated by the server-side code using Flask's template engine. The submit button allows users to submit the form and add a stock to their portfolio.

2. Quantittaive Momentum Strategy model
    * Brief explanation of momentum model: <br>
    The quantitative momentum model builds an investing strategy that selects the 50 US stocks from S&P 500 index with the best momentum metrics. From there, recommended trades for an equal-weight portfolio of these 50 stocks are calculated.<br>
        * Momentum stocks can be divided into 2 categories- High and low-quality. High-quality momentum stocks show slow and steady outperformance over long periods. Low-quality momentum stocks might only show momentum for a short time and then surge upwards. High-quality momentum stocks are preferred because low-quality momentum can often be caused by short-term news that is unlikely to be repeated in the future (such as an FDA approval for a biotechnology company). To identify high-quality momentum, we're going to build a strategy that selects stocks from the highest percentiles of 1-month price returns, 3-month price returns, 6-month price returns, and 1-year price returns. <br>

        * To give a more comprehensive overview, this is a python script that retrieves and processes financial metrics data of S&P 500 companies. It uses data from CSV files (another way is to make API call from Financial Modeling Prep) such as 1-year returns, 6-month returns, 3-month returns, and 1-month returns and stores these metrics in a pandas data frame. The data is then stored in a Python dictionary, then converted into a pandas data frame, and the columns are rearranged for further analysis. The data frame is sorted by the stocks' 4 metrics and drops all stocks outside the top 50. The user then enters the portfolio value. The HQM Score (which stands for high-quality momentum) is the momentum score that will be used to filter stocks in this investing strategy. The HQM Score will be the arithmetic mean of the 4 percentile scores we calculated in the last section. The number of shares to buy is then calculated for each stock in 50 selected stocks.

    * Flask implementaion of momentum model in flask app: <br>
     The flask implementation of this model defines two flask routes. The first route, /momentummodel, handles GET requests and returns a form in the momentum.html template. The second route, also /momentummodel, handles POST requests and processes the form data to generate a portfolio of momentum stocks. The portfolio is generated as follows:
        * Retrieve momentum metrics of 503 stocks in S&P 500 index by making API calls, but when I run it as part of a web application, the code is executed in a different environment with limited resources, causing the application to run slower. Since the project is developed using CS50 codespace, I could not change the timeout limit as a third party manages the server, and the settings are not adjustable. So I used CSV file to download tickers and momentum metrics. Reads in a CSV file named momentum_metrics_sp500.csv into a Pandas DataFrame called "mf".
        * Calculates the momentum percentiles for various metrics and stores the result in new columns in the mf DataFrame.
        * Selects the 50 best momentum stocks by sorting the DataFrame by a new 'HQM Score' column, which is the mean of the momentum percentiles.
        * Calculates the number of shares to buy for each stock based on the portfolio size entered in the form and the stock price.
        * Finally, the processed data frame is returned to the client.

    * HTML template of momentum model: <br>
    The HTML template "momentum.html" uses bootstrap, a CSS framework, for styling and has links to several pages, including a home page, a machine learning model to analyze a portfolio, and two quantitative investment strategies (momentum and value).
        * On the "Momentum Model" page, the user enters portfolio value in a form, and a message is displayed if there's an error in the input.
        * The template has a header, navigation bar, logo, and footer. The template's body contains text explaining the purpose of the "Momentum Model" page and the form for entering portfolio value.

3. Value Strategy: A brief explanation of the Value Strategy and its implementation.
    * Brief explanation of value model: <br>
    The quantitative value model builds an investing strategy that selects the 50 US stocks from S&P 500 index with the best value metrics. From there, recommended trades for an equal-weight portfolio of these 50 stocks are calculated. <br>
        * Every valuation metric has certain flaws. For example, the PE ratio doesn't work well with stocks with negative earnings. Similarly, stocks that buy back their shares are difficult to value using the PB ratio. A composite basket of valuation metrics is typically used to build robust quantitative value strategies. We will filter for stocks with the lowest percentiles on the following metrics- PE ratio, PB ratio, PS ratio, EV/EBITDA, and EV/GP. <br>
        * To give a more comprehensive overview, this is a python script that retrieves and processes financial metrics data of S&P 500 companies. It uses data such as PE ratio, PB ratio, PS Ratio, EV/EBITDA, and EV/FCF ratios from CSV file (another way is to make an API call from Financial Modeling Prep). These metrics are stored in a pandas data frame. The data is then stored in a Python dictionary, then converted into a pandas data frame, and the columns are rearranged for further analysis. The data frame is sorted by the stocks' 5 metrics and drops all stocks outside the top 50. The user then enters the portfolio value. The RV Score (which stands for Robust Value) is the value score that will be used to filter stocks in this investing strategy. The RV Score will be the arithmetic mean of the 4 percentile scores we calculated in the last section. The number of shares to buy is then calculated for each stock in 50 selected stocks.

    * Flask implementaion of value model in flask app <br>
    The flask implementation of this model defines two flask routes. The first route, /valuemodel, handles GET requests and returns a form in the value.html template. The second route, also /valuemodel, handles POST requests and processes the form data to generate a portfolio of value stocks. The portfolio is generated as follows:
        * Retrieve value metrics of 503 stocks in S&P 500 index by making API calls, but when I run it as part of a web application, the code is executed in a different environment with limited resources, causing the application to run slower. Since the project is developed using CS50 codespace, I could not change the timeout limit as a third party manages the server, and the settings are not adjustable. So I used CSV file to download tickers and value metrics. Reads in a CSV file named value_metrics_sp500.csv into a Pandas DataFrame called "vf".
        * Calculates the value percentiles for various metrics and stores the result in new columns in the vf DataFrame.
        * Selects the 50 best value stocks by sorting the DataFrame by a new 'RV Score' column, which is the mean of the value percentiles.
        * Calculates the number of shares to buy for each stock based on the portfolio size entered in the form and the stock price.
        * Finally, the processed data frame is returned to the client.

    * HTML template of value model <br>
    The HTML template "value.html" uses bootstrap, a CSS framework, for styling and has links to several pages, including a home page, a machine learning model to analyze a portfolio, and two quantitative investment strategies (momentum and value).
        * On the "Value Model" page, the user enters portfolio value in a form, and a message is displayed if there's an error in the input.
        * The template has a header, navigation bar, logo, and footer. The template's body contains text explaining the purpose of the "Value Model" page and the form for entering portfolio value.

#### Scope: <br>
This project's scope is to provide a fundamental solution to analyze stock portfolios efficiently using a robust platform. However, there is always room for improvement. Possible extensions for this project could include the following:
* Extending the stock environment from the US to other geographical locations.
* One can use Principal Component Analysis (PCA) to analyze investment portfolios further. The results of PCA can be incorporated into portfolio optimization models, allowing for more informed investment decisions.
* Instead of using a CSV file, the stock data can be retrieved using batch API calls (I have included the code for API calls in the comments of the app.py file).
* Switching to API makes causing the application to run slower. So one can use a production WSGI server instead of the development server (in this case, the CS50 codespace) in a production environment. We can use a production-ready WSGI server, such as uWSGI, Gunicorn, mod_wsgi, etc.
<br>

By continuing to develop and expand upon this project, we can [realize the objective of using quantitative and machine-learning concepts to make investment decisions. Feel free to contribute and take this project to new heights.
