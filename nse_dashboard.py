import yfinance as yf
import pandas as pd
import streamlit as st
from sklearn.svm import SVC 
import numpy as np
from nselib import capital_market, derivatives  # Assuming these libraries exist

# Additional import statements for matplotlib
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Load CSV data
df = pd.read_csv("D:\\DP project\\StockData.csv")

# Changes The Date column as index columns 
df.index = pd.to_datetime(df['Date']) 
df = df.drop(['Date'], axis='columns')  # drop The original date column 

# Create predictor variables 
df['Open-Close'] = df.Open - df.Close 
df['High-Low'] = df.High - df.Low 

# Store all predictor variables in a variable X 
X = df[['Open-Close', 'High-Low']] 
split_percentage = 0.8
split = int(split_percentage * len(df)) 

# Target variables 
y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0) 

# Train data set 
X_train = X[:split] 
y_train = y[:split] 

# Test data set 
X_test = X[split:] 
y_test = y[split:]

# Load SVM model and train it
svm_model = SVC()
svm_model.fit(X_train, y_train)  # Assuming X_train and y_train are defined

# Function to fetch SVM predictions for a given stock ticker
def get_svm_predictions(ticker):
    # Fetch historical stock data using yfinance
    stock_data = yf.download(ticker, start='2023-01-01', end='2024-01-01', progress=False)
    df = stock_data.copy()

    # Preprocess data
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low
    X = df[['Open-Close', 'High-Low']]

    # Predict signals using the trained SVM model
    df['Predicted_Signal'] = svm_model.predict(X)
    df['Return'] = df.Close.pct_change()
    df['Strategy_Return'] = df.Return * df.Predicted_Signal.shift(1)
    df['Cum_Ret'] = df['Return'].cumsum()
    df['Cum_Strategy'] = df['Strategy_Return'].cumsum()

    return df[['Close', 'Predicted_Signal', 'Return', 'Strategy_Return', 'Cum_Ret', 'Cum_Strategy']]
# Streamlit code for the dashboard
st.header('Indian Stock Financial Dashboard 2024')



instrument = st.sidebar.selectbox('Instrument Type', options=('NSE Equity Market', 'NSE Derivative Market', 'Candlestick Chart', 'Stock Market Prediction (SVM)'))

if instrument == 'NSE Equity Market':
    data_info = st.sidebar.selectbox('Data to Extract', options=('bulk_deal_data', 'block_deals_data', 'short_selling_data', 'bhav_copy_with_delivery',
                                                                 'bhav_copy_equities', 'equity_list', 'fno_equity_list', 'nifty50_equity_list', 'india_vix_data',
                                                                 'index_data', 'market_watch_all_indices', 'fii_dii_trading_activity'))

    if data_info in ('equity_list', 'fno_equity_list', 'market_watch_all_indices', 'nifty50_equity_list'):
        data = getattr(capital_market, data_info)()
        st.write(data)

    if data_info in ('bhav_copy_with_delivery', 'bhav_copy_equities'):
        date = st.sidebar.text_input('Date', '19-04-2024')
        data = getattr(capital_market, data_info)(date)
        st.write(data)

    if data_info in ('bulk_deal_data', 'block_deals_data', 'india_vix_data', 'short_selling_data'):
        period_ = st.sidebar.text_input('Period', '1M')
        data = getattr(capital_market, data_info)(period=period_)
        st.write(data)

elif instrument == 'NSE Derivative Market':
    data_info = st.sidebar.selectbox('Data to Extract', options=('expiry_dates_future', 'expiry_dates_option_index', 'fno_bhav_copy',
                                                                 'future_price_volume_data', 'nse_live_option_chain', 'option_price_volume_data',
                                                                 'participant_wise_open_interest', 'participant_wise_trading_volume', 'fii_derivatives_statistics'))

    if data_info in ('expiry_dates_future', 'expiry_dates_option_index'):
        data = getattr(derivatives, data_info)()
        st.write(data)

    if data_info in ('fii_derivatives_statistics', 'fno_bhav_copy', 'participant_wise_trading_volume', 'participant_wise_open_interest'):
        date = st.sidebar.text_input('Date', '19-04-2024')
        data = getattr(derivatives, data_info)(date)
        st.write(data)

    if data_info == 'future_price_volume_data':
        ticker = st.sidebar.text_input('Ticker', 'SBIN')
        type_ = st.sidebar.text_input('Instrument Type', 'FUTSTK')
        period_ = st.sidebar.text_input('Period', '1M')
        data = derivatives.future_price_volume_data(ticker, type_, period=period_)
        st.write(data)

    if data_info == 'option_price_volume_data':
        ticker = st.sidebar.text_input('Ticker', 'BANKNIFTY')
        type_ = st.sidebar.text_input('Instrument Type', 'OPTIDX')
        period_ = st.sidebar.text_input('Period', '1M')
        data = derivatives.option_price_volume_data(ticker, type_, period=period_)
        st.write(data)

    if data_info == 'nse_live_option_chain':
        ticker = st.sidebar.text_input('Ticker', 'BANKNIFTY')
        expiry_date = st.sidebar.text_input('Expiry Date', '19-04-2024')
        data = derivatives.nse_live_option_chain(ticker, expiry_date=expiry_date)
        st.write(data)
        
elif instrument == 'Candlestick Chart':
    st.sidebar.subheader('Candlestick Chart Settings')
    ticker_input = st.sidebar.text_input('Stock Ticker (e.g., TATASTEEL.NS)', 'TATASTEEL.NS')
    period_input = st.sidebar.selectbox('Period', options=['1mo', '3mo', '6mo', '1y'])
    chart_type = st.sidebar.selectbox('Chart Type', options=['Daily', 'Weekly'])

    # Fetch historical data
    stock = yf.Ticker(ticker_input)
    hist = stock.history(period=period_input)

    # Convert date index to Matplotlib date format
    hist['Date'] = pd.to_datetime(hist.index)
    hist['Date'] = hist['Date'].apply(mdates.date2num)

    # Plot candlestick chart based on the selected type
    if chart_type == 'Daily':
        st.subheader(f'{ticker_input} Candlestick Chart (Daily)')
        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed
        candlestick_ohlc(ax, hist[['Date', 'Open', 'High', 'Low', 'Close']].values, width=0.6, colorup='green', colordown='red')
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'{ticker_input} Candlestick Chart (Daily)')
        st.pyplot(fig)
    elif chart_type == 'Weekly':
        st.subheader(f'{ticker_input} Candlestick Chart (Weekly)')
        weekly_data = hist.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
        weekly_data['Date'] = pd.to_datetime(weekly_data.index)
        weekly_data['Date'] = weekly_data['Date'].apply(mdates.date2num)
        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed
        candlestick_ohlc(ax, weekly_data[['Date', 'Open', 'High', 'Low', 'Close']].values, width=0.6, colorup='green', colordown='red')
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'{ticker_input} Candlestick Chart (Weekly)')
        st.pyplot(fig)

elif instrument == 'Stock Market Prediction (SVM)':
    st.sidebar.subheader('Stock Market Prediction using SVM')
    ticker_input_svm = st.sidebar.text_input('Enter Stock Ticker Symbol (e.g., AAPL)', 'AAPL')
    if st.sidebar.button('Get SVM Predictions'):
        svm_predictions_df = get_svm_predictions(ticker_input_svm)
        stock_return_last_year = (svm_predictions_df['Close'][-1] / svm_predictions_df['Close'][0] - 1) * 100
        strategy_return = svm_predictions_df['Cum_Strategy'][-1]

        st.write(f"{ticker_input_svm} Stock Return Over Last 1 year - {stock_return_last_year:.2f}%")
        st.write(f"{ticker_input_svm} Strategy result - {strategy_return:.2f}%")

        # Plot Strategy Returns vs Original Returns for the company
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(svm_predictions_df.index, svm_predictions_df['Cum_Ret'], label='Original Returns')
        ax.plot(svm_predictions_df.index, svm_predictions_df['Cum_Strategy'], label='Strategy Returns')
        ax.set_xlabel('Date')
        ax.set_ylabel('Return')
        ax.set_title(f'{ticker_input_svm} - Strategy Returns vs Original Returns')
        ax.legend()
        st.pyplot(fig)