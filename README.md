# Project Overview
The purpose of this project is to determine when to take or exit a position on a financial instrument by training a machine learning model (neural network) on
OHLCV data from time bars for market instruments such as the S%P500 pulled from APIs, attempting to deploy the model and generalise on similar data. Labelling of training data is done based on three conditions labelling a datapoint as buy, sell, or hold:

1. For a datapoint, if the future close price exceeds an upper bound within a set time frame it is labelled as buy (1), this is done by specifying a timeframe (such as 5 days) and setting the upper bound using a Bollinger band.
2. For a datapoint, if the close price drops below the lower bound withint the set time frame it is labelled as sell (-1), the lower bound is a lower Bollinger band.
3. If the close price does not cross any bounds within the time frame it is labelled as hold (0).

The formulation used to generate the upper and lower Bollinger band is defined as:

$$
\text{Close Price} \pm n\sigma
$$

where:
1. $n$ is a scaling factor (typically 2).
2. $\sigma$ is the standard deviation of closing prices.

![Monte Carlo Result](Images/Labelling.png)

The figure above shows an exmaple of how I have established labelling thresholds, exceeding the green limit before any other boundaries results in a buy label (1), exceeding the red results in a sell label (-1) and exceeding the white results in a hold (0). It was commonly found that the hold signal (0) is particularly scarce across tested data nd in these cases the problem has been reduced to a binary classification problem of buy and sell.

# Data Scaling and Features
This project generally works with returns scaled in a Z-score normalisation style with the addition of some traditional technical indicators. Improvements would come in the form of using more robust, statistically sound and information rich features such as through the use of fractional differencing, with the inclusion of tests for feature importance.

# Future Work
Future work will focus on:
1. Methods to improve accuracy, these will likely revolve around addressing issues in data leakage, scaling and use of more robust features, and model architecture. The effect on performance when using different data structures, particularly event based sampling such as tick, volume or dollar bars should be explored. Alternative data should also be considered.
2. Developing robust custom backtests to evaluate out of sample performance.
3. Deployment to paper traiding to evaluate performance, ensuring the exact logic is tranlated into live trading.
