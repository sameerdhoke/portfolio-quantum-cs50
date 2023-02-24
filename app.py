from flask import Flask, render_template, request
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import matplotlib.pyplot as plt
from fuzzywuzzy import process
import math
import requests
import random
import numpy as np
from scipy import stats
from statistics import mean
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import datetime
# Defining the start and end dates for a time range that will be used to fetch stock data
end = datetime.datetime.now()
start = end - datetime.timedelta(days=3650)

app = Flask(__name__, static_folder='static')

# Setting the value of a configuration option for the Flask application
# The option TEMPLATES_AUTO_RELOAD is used to control whether Flask should automatically reload templates if they change on disk.
# The value "True" implies that Flask will watch the templates directory for changes and reload templates whenever they are modified.
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Creating an empty python list that wil be later used to store information about a stock portfolio
portfolio = []

# Flask route definition for the index page of the web application
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Flask route definition for a page that displays a form for entering information about a stock portfolio
@app.route('/mlmodel', methods=['GET'])
def show_ml_form():

    # Loading a csv file named sp_500_stocks.csv into a pandas DataFrame named stocks.
    stocks = pd.read_csv('sp_500_stocks.csv')
    # Converting the compName column of the stocks data frame into a list and assigns it to the options variable
    options = stocks['compName'].tolist()

    # Generating an HTML string containing a <datalist> element with <option> elements for each item in the options list
    html = '<datalist id="options">'
    for option in options:
        html += f'<option value="{option}">'
    html += '</datalist>'

    return render_template('ml.html', html=html)

# Flask route definition for a page that handles the processing of information entered in the form on the /mlmodel page
@app.route('/mlmodel', methods=['POST'])
def ml1():

    # Loading a csv file named sp_500_stocks.csv into a pandas DataFrame named stocks.
    stocks = pd.read_csv('sp_500_stocks.csv')
    # Converting the compName column of the stocks data frame into a list and assigns it to the options variable
    options = stocks['compName'].tolist()

    # Generating an HTML string containing a <datalist> element with <option> elements for each item in the options list
    html = '<datalist id="options">'
    for option in options:
        html += f'<option value="{option}">'
    html += '</datalist>'

    stock_name = request.form['stock_name']
    stock_quantity = request.form['stock_quantity']
    purchase_price_per_share = request.form['purchase_price_per_share']

    if 'done' in request.form:
        # Checking if the number of stocks entered is less than 5
        if len(portfolio) < 5:
            return render_template('ml.html', error="You need to enter at least 5 stocks. A portfolio less than 5 is a sign that the portofolio is undiversified and unbalanced", html=html)

        # Creating an empty pandas dataframe "ml_dataframe" with 3 columnsthat is store in thne list called "hqm_columns"
        hqm_columns = ['Ticker', 'Name of Stocks', 'Number of Shares']
        ml_dataframe = pd.DataFrame(columns = hqm_columns)

        # Looping through the items in the "portfolio" object and appending each item to the ",ml_dataframe" data frame
        for stock_name, stock_quantity, purchase_price_per_share in portfolio:
            ml_dataframe = ml_dataframe.append({'Ticker': 'x', 'Name of Stocks': stock_name, 'Number of Shares': stock_quantity, 'Purchase Price per Share': purchase_price_per_share}, ignore_index=True)

        # Iterating through each row of the "ml_dataframe" data frame using the iterrows() method.
        for index, row in ml_dataframe.iterrows():
            # Finding the best matching "compName" value in the "stocks" dataframe for the "Name of Stocks" in the current row of ml_dataframe using the process.extractOne method from the fuzzywuzzy library
            match = process.extractOne(row['Name of Stocks'], stocks['compName'])
            if match:
                # If a match is found, extract the corresponding "Ticker" value and "compName" value from the "stocks" dataframe.
                ticker = stocks[stocks['compName'] == match[0]]['Ticker'].values[0]
                name = stocks[stocks['compName'] == match[0]]['compName'].values[0]
                # Updating the "Ticker" value for the current row in the "ml_dataframe" dataframe with the extracted "Ticker" value.
                ml_dataframe.loc[index, 'Ticker'] = ticker

        # Creating an empty pandas DataFrame named stock_data, and then retrieving a list of all the ticker symbols from the "ml_dataframe" data frame using the "Ticker" indexer
        stock_data = pd.DataFrame()
        tickers = ml_dataframe["Ticker"]

        # Looping through the ticker symbols in the tickers list and retrieving the adjusted close price data for each ticker symbol
        for ticker in tickers:
            stock_data[ticker] = yf.download(ticker, start=start, end=end)['Adj Close']

        # Using the dropna method of the stock_data DataFrame to remove any rows with missing values
        stock_data.dropna(inplace=True)

        # Getting the latest date from stock_data table
        latest_data = stock_data.index[-1]

        # Initializing variables for current value and amount invested
        current_value = 0
        amount_invested = 0

        # Looping through the rows of ml_dataframen
        for index, row in ml_dataframe.iterrows():
            ticker = row['Ticker']
            num_shares = int(row['Number of Shares'])
            purchase_price = float(row['Purchase Price per Share'].replace(',', ''))
            # Getting the adj close price for the latest date from stock_data
            stock_price = float(stock_data.loc[latest_data, ticker])
            # Calculating current value and amount invested
            current_value += num_shares * stock_price
            amount_invested += num_shares * purchase_price
        current_value = round(current_value, 2)
        amount_invested = round(amount_invested, 2)

        # Calculating portfolio return
        portfolio_return = ((current_value - amount_invested) / amount_invested) * 100
        portfolio_return = round(portfolio_return, 2)

        # Normalizing the "stock_data"
        stock_data = stock_data / stock_data.iloc[0]

        # Creating an instance of MinMaxScaler and fitting it to the stock_data
        scaler = MinMaxScaler()
        scaler.fit(stock_data)
        # Transforming the stock_data by applying the scaling learned during the fit to the MinMaxScaler
        stock_data_normalized = scaler.transform(stock_data)

        # Creating a normalized DataFrame from the stock_data_normalized array
        stock_data_normalized = pd.DataFrame(stock_data_normalized)
        # Splitting the DataFrame into training and testing datasets
        train_data = stock_data_normalized.iloc[:int(len(stock_data_normalized) * 0.8)]
        test_data = stock_data_normalized.iloc[int(len(stock_data_normalized) * 0.8):]

        # Creating an instance of a linear regression model and fitting it to the training data.
        model = LinearRegression()
        model.fit(train_data.index.values.reshape(-1, 1), train_data)

        # Making predictions using the fitted linear regression model on the testing data
        predictions = model.predict(test_data.index.values.reshape(-1, 1))

        # Calculating various evaluation metrics to measure the performance of the linear regression model
        mae = mean_absolute_error(test_data, predictions)
        mae = round(mae, 3)
        mse = mean_squared_error(test_data, predictions)
        mse = round(mse, 3)
        rmse = math.sqrt(mse)
        rmse = round(rmse, 3)
        r2 = r2_score(test_data, predictions)
        r2 = round(r2, 3)

        # Interpretation of the evaluation metrics based on their values
        interpretation_metric = ""
        if mae < 0.5:
            interpretation_metric += "The mean absolute error is low, indicating that the model is making small errors on average. \n"
        else:
            interpretation_metric += "The mean absolute error is high, indicating that the model is making large errors on average. \n"
        if mse < 1:
            interpretation_metric += "The mean squared error is low, indicating that the model is making small squared errors on average. \n"
        else:
            interpretation_metric += "The mean squared error is high, indicating that the model is making large squared errors on average. \n"
        if rmse < 1:
            interpretation_metric += "The root mean squared error is low, indicating that the model is making small errors on average. \n"
        else:
            interpretation_metric += "The root mean squared error is high, indicating that the model is making large errors on average. \n"

        # Generating a string of interpretation for the results of a portfolio prediction evaluation
        interpret_rec = ""
        if mae < 0.5 and rmse < 1 and mse < 1:
            interpret_rec += "Based on the low mean absolute error, root mean squared error, and mean square error, it indicates that the predicted returns are likely to be accurate, stable, and the portfolio is well-balanced and well-diversified. So, continue with the current portfolio."
        elif mae > 0.5 and rmse > 1:
            interpret_rec += "Based on the high mean absolute error and root mean squared error, the predicted returns are not very accurate and the portfolio may not be well-diversified. So, re-evaluate your portfolio and make changes to improve its diversification."
        else:
            # mse >1
            interpret_rec += "Based on the high mean square error, the predicted returns are not very stable and the portfolio may not be well-balanced. So, re-evaluate the portfolio and make changes to improve its balance. "

        return render_template('ml.html', html=html, table=ml_dataframe.to_html(), mae=mae, mse=mse, rmse=rmse, r2=r2, interpretation_metric=interpretation_metric, interpret_rec=interpret_rec, current_value=current_value, amount_invested=amount_invested, portfolio_return=portfolio_return)

    else:

        # Checking if the inputs for "stock_quantity" and "purchase_price_per_share" are missing in a web form
        if len(request.form['stock_quantity']) == 0 or len(request.form['purchase_price_per_share']) == 0:
            return render_template('ml.html', error_m="Information missing! Please enter the required information for the stock.", html=html)

        # Checking if a stock name exists in the "compName" column of a dataframe "stocks"
        if stock_name in stocks['compName'].values:
            # Appending a tuple of the stock name, stock quantity , and purchase price per share to a list "portfolio"
            portfolio.append((stock_name, stock_quantity, purchase_price_per_share))

            # Creating an empty pandas dataframe "ml_dataframe" with 3 columnsthat is store in thne list called "hqm_columns"
            hqm_columns = ['Ticker', 'Name of Stocks', 'Number of Shares']
            ml_dataframe = pd.DataFrame(columns = hqm_columns)

            # Looping through each stock in the "portfolio" list and appends a new row to the "ml_dataframe" dataframe
            for stock_name, stock_quantity, purchase_price_per_share in portfolio:
                ml_dataframe = ml_dataframe.append({'Ticker': 'x',
                                                    'Name of Stocks': stock_name,
                                                    'Number of Shares': stock_quantity,
                                                    'Purchase Price per Share': purchase_price_per_share
                                                    },
                                                    ignore_index=True
                                                    )
            # Using iterrows() method to loop through each row of the "ml_dataframe" dataframe.
            for index, row in ml_dataframe.iterrows():
                # Finding the best matching "compName" value in the "stocks" dataframe for the "Name of Stocks" in the current row of ml_dataframe using the process.extractOne method from the fuzzywuzzy library
                match = process.extractOne(row['Name of Stocks'], stocks['compName'])
                if match:
                    # If a match is found, extract the corresponding "Ticker" value and "compName" value from the "stocks" dataframe.
                    ticker = stocks[stocks['compName'] == match[0]]['Ticker'].values[0]
                    name = stocks[stocks['compName'] == match[0]]['compName'].values[0]
                    # Updating the "Ticker" value for the current row in the "ml_dataframe" dataframe with the extracted "Ticker" value.
                    ml_dataframe.loc[index, 'Ticker'] = ticker

            return render_template('ml.html', html=html, table=ml_dataframe.to_html())

        else:
            error_msg = "Invalid stock name! Please enter a valid stock name from the S&P500 index."
            if error_msg:
                # If the "error_msg" variable has a value, pass it as a parameter to the "render_template" method along with the original HTML content "html". This will display the error message on the HTML page.
                return render_template('ml.html', error_msg=error_msg, html=html)
            else:
                # If the "error_msg" variable does not have a value, that is equal to None or empty, only the original HTML content "html" is passed to the "render_template" method.
                return render_template('ml.html', html=html)

# Flask route definition that handles GET requests to the '/valuemodel' endpoint.
@app.route('/valuemodel', methods=['GET'])
def show_value_form():
    return render_template('value.html')

# Flask route definition that handles POST requests to the '/valuemodel' endpoint
@app.route('/valuemodel', methods=['POST'])
def value1():

    if 'submit' in request.form:

        # Retrieving the value of the 'port_value' field from the request data sent by a client in a POST request to the '/valuemodel' endpoint
        port_value = request.form['port_value']

        """Retrieving tickers and stock names of 503 stocks in S&P 500 index via API calls, dividing the tickers list in chunks, and downloading stock price, P/E, P/B, P/S, EV/EBITDA, EV/FCF data vial API calls"""
        # stocks_url = "https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={api_key}"
        # data = requests.get(stocks_url).json()
        # tick = [d['symbol'] for d in data]
        # stock_names = [d['name'] for d in data]

        # def chunks(lst, n):
        #     for i in range(0, len(lst), n):
        #         yield lst[i:i + n]
        # symbol_groups = list(chunks(tick, 5))
        # tickers = []
        # for i in range(0, len(symbol_groups)):
        #     tickers.append(','.join(symbol_groups[i]))

        # api_key = not disclosed for privacy purpose
        # financial_dir = {}
        # for ticker in tickers:
        #     symbols = ticker.split(',')
        #     for symbol in symbols:
        #         temp_dir = {}
        #         try:
        #             url = "https://financialmodelingprep.com/api/v3/company-key-metrics/"+symbol+"?apikey={}".format(api_key)
        #             page = requests.get(url)
        #             if page.status_code != 200:
        #                 print(f"Error occured for ticker {ticker} with status code {page.status_code} and error {page.content}")
        #                 continue
        #             fin_dir = page.json()
        #             for key,value in fin_dir["metrics"][0].items():
        #                 if key in ["Stock Price", "PE ratio","PB ratio", "Price to Sales Ratio","Enterprise Value over EBITDA","EV to Free cash flow"]:
        #                     temp_dir[key] = value
        #             url = "https://financialmodelingprep.com/api/v3/enterprise-value/"+symbol+"?apikey={}".format(api_key)
        #             page = requests.get(url)
        #             if page.status_code != 200:
        #                 print(f"Error occured for ticker {ticker} with status code {page.status_code} and error {page.content}")
        #                 continue
        #             fin_dir = page.json()
        #             for key,value in fin_dir["enterpriseValues"][0].items():
        #                 if key in ["Stock Price", "PE ratio","PB ratio", "Price to Sales Ratio","Enterprise Value over EBITDA","EV to Free cash flow"]:
        #                     temp_dir[key] = value
        #         except KeyError:
        #             temp_dir["PE ratio"] = None
        #             temp_dir["PB ratio"] = None
        #             temp_dir["Price to Sales Ratio"] = None
        #             temp_dir["Enterprise Value over EBITDA"] = None
        #             temp_dir["EV to Free cash flow"] = None
        #         financial_dir[symbol] = temp_dir
        # pd.options.display.max_rows = 20
        # combined_financials = pd.DataFrame(financial_dir)

        """ When I ran the code as a standalone file (with API calls), it was executed in a single environment with all the resources available to the system.
        However, when I run it as part of a web application, the code is executed in a different environment (e.g., a container or a virtual environment) with limited resources, causing the application to run slower.
        Additionally, the API calls cause slow performance.
        When I used 'requests' package to make API calls to the financialmodelingprep.com API, this package runs synchronously, which means that it will wait for a response from the API before
        it moves on to the following line of code, causing delays and slowing down the program.
        I configured the server settings to wait before timing out, but the program took too long to execute.
        Since I'm using CS50 codespace, I could not change the timeout limit as a third party manages the server, and the settings are not adjustable.
        One way to work around this is to break the ticker list into smaller chunks and make API calls in smaller groups instead of making all the calls simultaneously and ensure that the code is not too resource-intensive.
        Due to the above complication, I used data from a CSV file. """

        # Reading the contents of the file value_metrics_sp500.csv using the pd.read_csv function from the Pandas library and stores the resulting data in a Pandas DataFrame named vf
        vf = pd.read_csv('value_metrics_sp500.csv', encoding='latin1')

        # Calculating Value Percentiles
        metrics = { 'Price-to-Earnings Ratio': 'PE Percentile', 'Price-to-Book Ratio':'PB Percentile', 'Price-to-Sales Ratio': 'PS Percentile', 'EV/EBITDA':'EV/EBITDA Percentile', 'EV/FCF':'EV/FCF Percentile'}
        # Iterating through each row in the DataFrame vf using the .index attribute, which is a list of row labels
        for row in vf.index:
            #  Iterating over the keys in the metrics dictionary and updating the values in the Pandas DataFrame vf accordingly
            for metric in metrics.keys():
                vf.loc[row, metrics[metric]] = stats.percentileofscore(vf[metric], vf.loc[row, metric])/100

        # The value of each cell in the 'RV Score' column is calculated by taking the mean of the values in the specified set of columns
        vf['RV Score'] = vf[list(metrics.values())].mean(axis=1)

        # Selecting the 50 Best Value Stocks
        vf.sort_values(by = 'RV Score', inplace = True)
        vf = vf[:50]
        vf.reset_index(drop = True, inplace = True)

        # Calculating the Number of Shares to Buy
        port_value = request.form['port_value']
        if not port_value:
            return "Invalid input. Please enter a valid number for the portfolio value."
        try:
            portfolio_size = float(port_value)
        except ValueError:
            return "Invalid input. Please enter a valid number for the portfolio value."

        # Checking if the value stored in port_value is a valid positive number by using the isdigit() method
        if port_value.isdigit():
            portfolio_size = float(port_value)
            # Retrieving the value entered in the form field named port_value from the HTTP request sent by the client
            port_value = request.form['port_value']

            # Calculating the number of shares to buy for each stock in the portfolio and stores the results in a new column named 'Number of Shares to Buy' in the vf DataFrame
            vf['Number of Shares to Buy'] = (portfolio_size / len(vf.index)) / vf['Price']
            vf['Number of Shares to Buy'] = vf['Number of Shares to Buy'].apply(lambda x: math.floor(x))
        else:
            return "Please enter a valid number"

        # Rounding the values in multiple columns of the vf DataFrame to two decimal places
        vf[["Price", "Price-to-Earnings Ratio", "PE Percentile", "Price-to-Book Ratio", "PB Percentile","Price-to-Sales Ratio", "PS Percentile", "EV/EBITDA", "EV/EBITDA Percentile",
        "EV/FCF", "EV/FCF Percentile"]] = vf[["Price", "Price-to-Earnings Ratio", "PE Percentile", "Price-to-Book Ratio", "PB Percentile", "Price-to-Sales Ratio", "PS Percentile", "EV/EBITDA",
        "EV/EBITDA Percentile", "EV/FCF", "EV/FCF Percentile"]].apply(lambda x: round(x,2))

        value_df = vf
        vf = vf.reindex(columns=["Name", "Ticker", "Price", "Number of Shares to Buy"])
        # Converting the values in the 'Number of Shares to Buy' column of the vf DataFrame from float to integer and storing the result in the same column
        vf["Number of Shares to Buy"] = vf["Number of Shares to Buy"].astype(int)
        final_value_df = vf

        return render_template('value.html', port_value=port_value, value_table=value_df.to_html(), final_value_table=final_value_df.to_html())

    else:

        port_value = request.form['port_value']

        # Checking if the value stored in port_value is a positive number. If it is not, an error message is returned to the user.
        if not port_value.isdigit() or int(port_value) <= 0 :
            return render_template('value.html', error="Invalid entry: The portfolio value must be a positive integer or a float")

        return render_template('value.html', port_value = port_value)

# Defining a new endpoint for a Flask web application that is accessed using a GET request to the URL "/momentummodel"
@app.route('/momentummodel', methods=['GET'])
def show_momentum_form():
    return render_template('momentum.html')

# Defining another endpoint for a Flask web application that is accessed using a POST request to the URL "/momentummodel"
@app.route('/momentummodel', methods=['POST'])
def momentum1():

    if 'submit' in request.form:

        # Retrieving the value of the port_v field from the form data in the POST request
        port_v = request.form['port_v']

        """Retrieving tickers and stock names of 503 stocks in S&P 500 index via API calls, dividing the tickers list in chunks, and downloading one-year, six-month, three-month, one-month returns data vial API calls"""
        # stocks_url = "https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={api_key}"
        # data = requests.get(stocks_url).json()
        # tick = [d['symbol'] for d in data]
        # stock_names = [d['name'] for d in data]

        # def chunks(lst, n):
        #     for i in range(0, len(lst), n):
        #         yield lst[i:i + n]
        # symbol_groups = list(chunks(tick, 5))
        # tickers = []
        # for i in range(0, len(symbol_groups)):
        #     tickers.append(','.join(symbol_groups[i]))

        # api_key = not disclosed for privacy purpose
        # financial_dir = {}
        # for ticker in tickers:
        #     symbols = ticker.split(',')
        #     for symbol in symbols:
        #         temp_dir = {}
        #         try:
        #             url = "https://financialmodelingprep.com/api/v3/stock-price-change/"+symbol+"?apikey={}".format(api_key)
        #             page = requests.get(url)
        #             if page.status_code != 200:
        #                 print(f"Error occured for ticker {ticker} with status code {page.status_code} and error {page.content}")
        #                 continue
        #             fin_dir = page.json()
        #             for key,value in fin_dir[0].items():
        #                 if key in ["price", "1Y", "6M", "3M", "1M"]:
        #                     temp_dir[key] = value
        #             url = base_url+"/quote-short/"+symbol+"?apikey={}".format(api_key)
        #             page = requests.get(url)
        #             if page.status_code != 200:
        #                 print(f"Error occured for ticker {ticker} with status code {page.status_code} and error {page.content}")
        #                 continue
        #             fin_dir = page.json()
        #             for key,value in fin_dir[0].items():
        #                 if key in ["price", "1Y", "6M", "3M", "1M"]:
        #                     temp_dir[key] = value
        #         except KeyError:
        #             temp_dir["1Y"] = None
        #             temp_dir["6M"] = None
        #             temp_dir["3M"] = None
        #             temp_dir["1M"] = None
        #         financial_dir[symbol] = temp_dir
        # combined_financials = pd.DataFrame(financial_dir)

        """ When I ran the code (with API calls) as a standalone file, it was executed in a single environment with all the resources available to the system.
        However, when I run it as part of a web application, the code is executed in a different environment (e.g., a container or a virtual environment) with limited resources, causing the application to run slower.
        Additionally, the API calls cause slow performance.
        When I used 'requests' package to make API calls to the financialmodelingprep.com API, this package runs synchronously, which means that it will wait for a response from the API before
        it moves on to the following line of code, causing delays and slowing down the program.
        I configured the server settings to wait before timing out, but the program took too long to execute.
        Since I'm using CS50 codespace, I could not change the timeout limit as a third party manages the server, and the settings are not adjustable.
        One way to work around this is to break the ticker list into smaller chunks and make API calls in smaller groups instead of making all the calls simultaneously and ensure that the code is not too resource-intensive.
        Due to the above complication, I used data from a CSV file. """

        mf = pd.read_csv('momentum_metrics_sp500.csv', encoding='latin1')

        # Calculating Momentum Percentiles
        time_periods = ['One-Year', 'Six-Month', 'Three-Month', 'One-Month']

        # Using a nested for loop to iterate over the rows of the DataFrame mf and over the values in the time_periods list
        for row in mf.index:
            for time_period in time_periods:
                # Calculating the percentile rank of a given time period's stock return for a given row in the DataFrame "mf"
                mf.loc[row, f'{time_period} Return Percentile'] = stats.percentileofscore(mf[f'{time_period} Price Return'], mf.loc[row, f'{time_period} Price Return'])/100

        # Calculating the HQM Score
        for row in mf.index:
            momentum_percentiles = []
            # Creating a momentum_percentiles list and appending the percentile rank of the time period's stock return for each time period in the list "time_periods"
            for time_period in time_periods:
                momentum_percentiles.append(mf.loc[row, f'{time_period} Return Percentile'])
            # The value at the specified row and column "HQM Score" is being set to the mean of the values in the list "momentum_percentiles".
            mf.loc[row, 'HQM Score'] = mean(momentum_percentiles)

        # Sorting in descending order based on the "HQM Score" column
        mf = mf.sort_values(by = 'HQM Score', ascending = False)
        # Only the top 50 rows are being kept
        mf = mf[:50]
        # The index of dataframe is being reset and the original index is being dropped
        mf.reset_index(drop=True, inplace=True)

        # Calculating the Number of Shares to Buy
        port_v = request.form['port_v']

        # Conditional statement checking for the validity of the input port_v
        if not port_v:
            message = "Invalid input. Please enter a valid number for the portfolio value."
            return render_template("momentum.html", message=message)

        # The conversion to float is done inside a try-except block to catch a potential ValueError exception, which is raised when the string cannot be converted to a float
        try:
            portfolio_size = float(port_v)
        except ValueError:
            message = "Invalid input. Please enter a valid number for the portfolio value."
            return render_template("momentum.html", message=message)

        # Checking if the input port_v is a digit. If the input is a digit, then convert it to a float type and stores the result in the variable portfolio_size
        if port_v.isdigit():
            portfolio_size = float(port_v)
            port_v = request.form['port_v']

            # Calculating the amount of money to allocate to each stock by dividing portfolio_size by the number of stocks and dividing again by the stock price
            mf['Number of Shares to Buy'] = (portfolio_size / len(mf.index)) / mf['Price']
            # Rounding the number of shares to the nearest integer using the math.floor function, which rounds down to the nearest whole number
            mf['Number of Shares to Buy'] = mf['Number of Shares to Buy'].apply(lambda x: math.floor(x))
        else:
            return "Please enter a valid number"

        # Rounding the values in specific columns of the pandas DataFrame "mf" to 2 decimal places
        mf[["Price", "Number of Shares to Buy", "One-Year Price Return", "One-Year Return Percentile", "Six-Month Price Return", "Six-Month Return Percentile", "Three-Month Price Return",
        "Three-Month Return Percentile", "One-Month Price Return", "One-Month Return Percentile"]] = mf[["Price", "Number of Shares to Buy", "One-Year Price Return", "One-Year Return Percentile",
        "Six-Month Price Return", "Six-Month Return Percentile", "Three-Month Price Return", "Three-Month Return Percentile", "One-Month Price Return", "One-Month Return Percentile"]].apply(lambda x: round(x,2))

        momentum_df = mf
        mf = mf.reindex(columns=["Name", "Ticker", "Price", "Number of Shares to Buy"])
        # Converting the values in the "Number of Shares to Buy" column of the pandas DataFrame "mf" to integers
        mf["Number of Shares to Buy"] = mf["Number of Shares to Buy"].astype(int)
        final_momentum_df = mf

        return render_template('momentum.html', port_v=port_v, momentum_table=momentum_df.to_html(), final_momentum_table=final_momentum_df.to_html())

    else:

        port_v = request.form['port_v']

        # Checking if the value stored in port_v is a positive number
        if not port_v.isdigit() or int(port_v) <= 0 :
            return render_template('momentum.html', error="Invalid entry: The portfolio value must be a positive integer or a float")

        return render_template('momentum.html', port_v = port_v)