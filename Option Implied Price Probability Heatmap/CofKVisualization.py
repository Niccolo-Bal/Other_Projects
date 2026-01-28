import option_prob_heatmap as oph
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
import statsmodels.api as sm
import matplotlib.pyplot as plt

def main():
    testTickerStr = oph.getValid(input("Enter ticker symbol: "), type='ticker')
    if testTickerStr is None:
        return
    optionType = oph.getValid(input("Enter option type (call/put): "), validOptions=['call', 'put'])
    riskFreeRate = oph.getValid(input("Risk-free rate (e.g., 0.04 for 4%): "), type='float')
    graphDate = oph.getValid(input("Enter graph date (YYYY-MM-DD) or leave blank for most recent: "), 
                             validOptions=list(yf.Ticker(testTickerStr).options) + [''])
    if graphDate == '':
        graphDate = yf.Ticker(testTickerStr).options[0]
    testData = oph.get_data(testTickerStr, lag='day', periods=1000, optionType=optionType)
    ##
    ticker = yf.Ticker(testTickerStr)
    try:
        tickerPriceLabel = ticker.info['lastPrice']
    except: 
        tickerPriceLabel = ticker.history(period="1d")['Close'].iloc[0]
    graphDate_dt = pd.to_datetime(graphDate)
    dateSubset = testData[testData['expirationDate'] == graphDate_dt]
    expTime = graphDate_dt.tz_localize("UTC")
    nowTime = pd.Timestamp.now(tz="UTC")
    dateSubset = dateSubset.sort_values("strike") 
    dateSubset = dateSubset.drop_duplicates("strike", keep="last") 
    T = (expTime - nowTime).total_seconds() / (365 * 24 * 3600)
    K = dateSubset['strike'].values    
    exp_ranges = testData.groupby("expirationDate")["strike"].agg(["min","max"])
    K_min = exp_ranges["min"].max()
    K_max = exp_ranges["max"].min()
    K_grid = np.linspace(K_min, K_max, 400)     
    C = dateSubset['middlePrice'].values
    r = riskFreeRate
    if optionType == 'put': 
        C = C - K * np.exp(-r * T) + tickerPriceLabel
    C_interp_func = interp1d(K, C, kind='linear', bounds_error=False, fill_value='extrapolate')
    C_on_grid = C_interp_func(K_grid)
    s = len(K_grid) * np.var(C_on_grid) * 0.001
    # Fit spline on the standardized grid (prevents pdf getting cut off)
    CofK = sm.nonparametric.lowess(exog=K_grid, endog=C_on_grid, frac=0.2)
    CofK = UnivariateSpline(CofK[:,0], CofK[:,1], k=5, s=s)
    d2C = CofK.derivative(n=2)(K_grid)
    pdf = np.exp(r * T) * d2C
    pdf = np.maximum(pdf, 0)
    pdf = pdf / np.trapezoid(pdf, K_grid)
    pdf = oph.forceUnimodality(pdf, K_grid)
    ##
    pdf_plot = pdf * dateSubset['strike'].max() / (2 * pdf.max())
    plt.scatter(K, C, color='blue', label='Data points', zorder=2)
    plt.plot(K_grid, CofK(K_grid), color='red', label='Spline', linewidth=2)
    plt.plot(K_grid, pdf_plot, color='green', label='PDF', linewidth=1)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
