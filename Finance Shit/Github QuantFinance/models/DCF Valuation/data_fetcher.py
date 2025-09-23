import yfinance as yf
import pandas as pd
import numpy as np

def get_ticker_data(str_ticker):
    yf_ticker = yf.Ticker(str_ticker)
    df = yf_ticker.history(period='max')


