import pandas as pd
import yfinance as yf

def ticker_dataframe(tickers): # Converts a list of tickers to a dataframe of their yf info
    data = {}
    for ticker in tickers:
        try:
            ticker_info = yf.Ticker(ticker).info
            data[ticker] = ticker_info
        except Exception as e:
            print('Error fetching data for {}: {}'.format(ticker, e))
    tdf = pd.DataFrame(data)
    rankings = pd.DataFrame([range(1, len(tdf.columns) + 1)], columns=tdf.columns, index=['rank'])
    tdf = pd.concat([rankings, tdf], axis=0)
    return tdf
top_100_tickers = pd.read_excel('TICKERS.xlsx')['Tickers'].tolist()
top_100_df = ticker_dataframe(top_100_tickers).T
# top_100_df.to_csv('top_100.csv', index=True)
