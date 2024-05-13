import yfinance as yf
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

stock = yf.Ticker("TATASTEEL.NS")
hist = stock.history(period='12mo')

print(hist)


mpf.plot(hist, type='candle', title="TATASTEEL Candlestick Chart (Daily)", style='yahoo', figscale = 2.0, figratio = (1, 0.3))


weekly_data = hist['Close'].resample('W').ohlc()

mpf.plot(weekly_data, type='candle', title="TATASTEEL Candlestick Chart (Weekly)", style='yahoo', figscale = 2.0, figratio = (1, 0.3))
