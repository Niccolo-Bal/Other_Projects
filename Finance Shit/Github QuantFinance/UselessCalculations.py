import pandas as pd
import yfinance as yf
import numpy as np
import contextlib
import os

top_industry_tickers = pd.read_csv('company_industries.csv', index_col=0)
top_industry_tickers = top_industry_tickers.squeeze()
def stock_funny_ratio(TICK_):
    try:
        TICK = TICK_.upper()
        funny_industry  = yf.Ticker(TICK).info.get('industryKey')
        TICK = yf.Ticker(TICK)
        industry_ratios = []
        for industry_company in top_industry_tickers[top_industry_tickers == funny_industry].dropna().index:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stderr(devnull):
                    try:
                        ratio = (yf.Ticker(industry_company).info.get('enterpriseToEbitda'))
                        if ratio is not None:
                            industry_ratios.append(ratio)
                    except:
                        pass
        ratio_z = np.abs((industry_ratios - np.mean(industry_ratios)) / np.std(industry_ratios))
        industry_ratios = [industry_ratios[i] for i in range(len(industry_ratios)) if ratio_z[i] < 2]
        comp_value = np.mean(industry_ratios) / TICK.info['enterpriseToEbitda']
        low_analyist_target = (TICK.analyst_price_targets['mean']) * 0.9
        if comp_value > 0:
            return (comp_value * low_analyist_target) / TICK.fast_info['lastPrice']
        elif TICK.info['enterpriseToEbitda'] <= 0:
            if np.mean(industry_ratios) <= 0: return "SC/SI"
            else: return "SC"
        else: return "SI"
    except:
        print('stock_funny_ratio() failed for \"{}\"'.format(TICK_))
        return None

top_100_tickers = pd.read_excel('TICKERS.xlsx')['Tickers'].tolist()

# for ticker in top_100_tickers:
    # print(ticker + ":" + str(stock_funny_ratio(ticker)))
# print(stock_funny_ratio('GOOG'))
print(stock_funny_ratio('META'))