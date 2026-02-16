# This script aims to collect data from financial instruments and label what positions should have been taken at each datapoint (S&P500 data used as example).
# Symmetrical Bollinger bands are used to determine the stop loss and take profit limits, these are computed based on the historical volatility calculated for a determined time window.
# Based on the window a time limit is set for the position to be determined.
# If a future price within the window exceeds the take profit limit, a long position label is given (1) and a short position label (-1) is given if the price drops below the stop loss.
# A sideways label (0) is allocated if future prices do not pass these limits and the time exceeds the window time limit.
# This data can be used to train ML models, and converted into a binary problem to boost accuracy in the case of scarce sideways cases (0 labels).
# Data features are generated and scaled.

# Library imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
import talib
import requests

plt.style.use('dark_background') # Use dark background for plots

end = dt.datetime.now() # End point for data collection
years = 5 # No. of years for data collection
start = end - dt.timedelta(days=years*365) # Start point for data collection
plot = True # Show plots if True, displays bounds for each data point
binary_problem = True # Reduce to a binary problem for classification
split = 0.8 # Train/Test data split
window = 5 # Window for calculating bands
bollinger_factor = 100 # Defines width of bollinger bands, a higher value increases the distance between the take profit/stop loss limits and the close price

# S&P500 tickers for data extraction
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies" # URL containing list of S&P 500 companies

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)
tables = pd.read_html(response.text)
sp500 = tables[0]

# Tickers list
tickers = sp500["Symbol"].to_list()

tickers = [t.replace(".", "-") for t in tickers] # Re-format for yahoo finance

# Download data and split into train and test set
def get_data(ticker, start, end, split):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, threads=False)
    df = pd.DataFrame(data)
    split_idx = int(len(df)*split)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    return df_train, df_test

# Label data based on bands exceeded
def labelling(df, window, plot, ticker, bollinger_factor):
    # Create plot
    if plot==True:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(f'Labelling of {ticker} Close Price', fontsize=16)
        axes[0].plot(df.index, df['Close'], label='Close Price')
        axes[1].plot(df.index, df['Close'], label='Close Price')
    # Initialise arrays
    bollinger_high = np.zeros(len(df['Close'].values))
    bollinger_low = np.zeros(len(df['Close'].values))
    price = np.zeros(len(df['Close'].values))
    labels = np.zeros(len(df))
    # Slide window across data and calculate bands
    for i in range(0,len(df['Close'].values)):
        if len(df.index[0:i]) > window and i + window < len(df.index):
            std = float(df['Close'].iloc[i-window:i].std())
            bollinger_high[i] = float(df['Close'].iloc[i]) + bollinger_factor * std
            bollinger_low[i] = float(df['Close'].iloc[i]) - bollinger_factor * std
            price[i] = float(df['Close'].iloc[i])
            # Find band locations for plot
            high_band_location = df.index[i:i+window+1]
            high_value = np.full(len(high_band_location), bollinger_high[i])
            low_band_location = df.index[i:i+window+1]
            low_value = np.full(len(low_band_location), bollinger_low[i])
            price_location = df.index[i:i+window+1]
            price_value = np.full(len(price_location), df['Close'].values[i])
            time_band_value_front = np.linspace(df['Close'].iloc[i] - bollinger_factor * std, df['Close'].iloc[i] + bollinger_factor * std, 10)
            time_band_location_front = np.full(len(time_band_value_front), df.index[i+window])
            time_band_value_back = np.linspace(df['Close'].iloc[i] - bollinger_factor * std, df['Close'].iloc[i] + bollinger_factor * std, 10)
            time_band_location_back = np.full(len(time_band_value_back), df.index[i])
            # Label data points
            future_prices = df['Close'].values[i+1:i+window+1]
            if np.any(future_prices > bollinger_high[i]):
                labels[i] = 1
            elif np.any(future_prices < bollinger_low[i]):
                labels[i] = -1
            else:
                labels[i] = 0
            # Plot bands
            if plot == True:
                axes[0].plot(high_band_location, high_value, '--', color='green', label='High Bollinger Band (Take Profit)' if i == window + 1 else '')
                axes[0].plot(low_band_location, low_value, '--', color='red', label='Low Bollinger Band (Stop Loss)' if i == window + 1 else '')
                axes[0].plot(price_location, price_value, '--', color='white', label='Close Price' if i == window + 1 else '')
                axes[0].plot(time_band_location_front, time_band_value_front, '--', color='white', label='Front Time Band' if i == window + 1 else '')
                axes[0].plot(time_band_location_back, time_band_value_back, '--', color='white', label='Back Time Band' if i == window + 1 else '')
    # Show plot
    if plot == True:
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel(f'{ticker} Close Price')
        axes[0].legend()
        plt.tight_layout()
        plt.show()
    return labels

# Generate features for ML 
def feature_generation(df, labels, window, binary_problem):
    # Generate features
    close = df['Close'].values.ravel()
    close_returns = df['Close'].pct_change()
    high_returns = df['High'].pct_change()
    low_returns = df['Low'].pct_change()
    open_returns = df['Open'].pct_change()
    volume_returns = df['Volume'].pct_change()
    sma = df['Close'].rolling(window).mean()
    ema = talib.EMA(close, timeperiod=window)
    momentum = talib.MOM(close, timeperiod=window)
    rsi = talib.RSI(close, timeperiod=window)
    volume_mean = df['Volume'].ewm(span=window).mean()
    price_volatility = close_returns.ewm(span=window, adjust=False).std()
    volume_volatility = df['Volume'].ewm(span=window).std()
    # Assign to columns
    df['Close Returns'] = close_returns #6 (-1 for numpy array)
    df['High Returns'] = high_returns #7
    df['Low Returns'] = low_returns #8
    df['Open Returns'] = open_returns #9
    df['Volume Returns'] = volume_returns #10
    df["Exponential Moving Average"] = ema #11
    df['Simple Moving Average'] = sma #12
    df["Relative Strength Index"] = rsi #13
    df["Momentum"] = momentum #14
    df['Volume Mean'] = volume_mean #15
    df['Price Volatility'] = price_volatility #16
    df['Volume Volatility'] = volume_volatility #17
    data = df.to_numpy()
    # Eliminate sideways cases
    if binary_problem == True:
        mask = np.isfinite(data).all(axis=1)
        mask &= np.isfinite(labels)
        data = data[mask]
        labels = labels[mask]
        # Labelling form for binary problem
        for i in range(0, len(labels)):
            if labels[i] == -1:
                labels[i] += 1
    return data, labels

# Scale data
def scaling(data):
    data[:,5] = data[:,5] / data[:,15] #Scaled close returns
    data[:,6] = data[:,6] / data[:,15] #Scaled high returns
    data[:,7] = data[:,7] / data[:,15] #Scaled low returns
    data[:,8] = data[:,8] / data[:,15] #Scaled open returns
    data[:,9] = data[:,9] / data[:,16] #Scaled volume returns
    data[:,10] = (data[:,0] - data[:,10]) / data[:,15] #Scaled exponential moving average
    data[:,12] = (data[:,12] - 50) / 50 #Scaled relative strength index
    data[:,13] = data[:,13] / data[:,15] #Scaled momentum
    return data

# Construct a dataset made from various tickers 
def data_set(tickers, start, end, split, window):
    # Arrays to concatenate data
    chunks_X_train = []
    chunks_Y_train = []
    chunks_X_test = []
    chunks_Y_test = []
    for i in range(0, len(tickers)):
        df_train, df_test = get_data(tickers[i], start, end, split) # Download data
        labels_train = labelling(df_train, window, True, tickers[i], bollinger_factor) # Label training data
        data_train, Y_train = feature_generation(df_train, labels_train, window, True) # Generate features for the training data
        labels_test = labelling(df_test, window, True, tickers[i], bollinger_factor) # Label test data
        data_test, Y_test = feature_generation(df_test, labels_test, window, True) # Generate features for the test data
        X_train = scaling(data_train) # Scale training data
        X_test = scaling(data_test) # Scale test data
        # Add individual ticker data to their respective array
        chunks_X_train.append(X_train)
        chunks_X_test.append(X_test)
        chunks_Y_train.append(Y_train)
        chunks_Y_test.append(Y_test)
    # Concatenate data for each ticker to construct full dataset
    X_data_train = np.concatenate(chunks_X_train, axis=0)
    X_data_test = np.concatenate(chunks_X_test, axis=0)
    Y_data_train = np.concatenate(chunks_Y_train, axis=0)
    Y_data_test = np.concatenate(chunks_Y_test, axis=0)
    return X_data_train, Y_data_train, X_data_test, Y_data_test

# Save data
def save_data(X_data, Y_data, filepath='data.npz'):
    """Save preprocessed data and labels to a compressed numpy file."""
    np.savez_compressed(filepath, X_data=X_data, Y_data=Y_data)
    print(f"Data saved to {filepath}")

# Construct dataset and save data
X_data_train, Y_data_train, X_data_test, Y_data_test = data_set(tickers, start, end, split, window)
save_data(X_data_train, Y_data_train, filepath='Training_Data_S&P500.npz')
save_data(X_data_test, Y_data_test, filepath='Test_Data_S&P500.npz')