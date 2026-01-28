import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import textwrap as tw
import statsmodels.api as sm
from scipy.interpolate import UnivariateSpline, interp1d
from yfinance import ticker

def getValid(inputVal, validOptions = ['y', 'n'], type='str'):
    if type == 'str':
        while inputVal.lower() not in validOptions:
            inputVal = input(f"(Valid inputs are: {', '.join(validOptions)}): ")
        return inputVal.lower()
    if type == 'int':
        while True:
            try:
                inputVal = int(inputVal)
                return inputVal
            except ValueError:
                inputVal = input("Please enter a valid integer: ")
    if type == 'float':
        while True:
            try:
                inputVal = float(inputVal)
                return inputVal
            except ValueError:  
                inputVal = input("Please enter a valid number: ")
    if type == 'ticker':
        while True:
            try:
                ticker = yf.Ticker(inputVal)
                testTicker = ticker.info
                return inputVal.upper()
            except:
                try: 
                    testSPY = yf.Ticker("SPY").info
                    inputVal = input("Please enter a valid ticker symbol: ")
                except:
                    print("Error retrieving yfinance data. Please check your internet connection.")
                    return None

def forceUnimodality(values, grid): # Take largest set of nonzero values
    values_nonzero = values != 0
    padded_values = np.concatenate([[False], values_nonzero, [False]])
    diff = np.diff(padded_values.astype(int))
    start_indices = np.where(diff == 1)[0]
    end_indices = np.where(diff == -1)[0]
    if len(start_indices) <= 1:
        return values
    lengths = end_indices - start_indices
    longest_idx = np.argmax(lengths)
    longest_start = start_indices[longest_idx]
    longest_end = end_indices[longest_idx]
    trimmed_y = np.zeros_like(values)
    trimmed_y[longest_start:longest_end] = values[longest_start:longest_end]
    return trimmed_y

def inputSettings(): # Retrieve lag, periods, option type from user
    lag = getValid(input("Lag (day/wk/mon): "), ["day","wk", "mon"])
    periods = getValid(input("Number of periods to consider (e.g., 30): "), type='int')
    optionType = getValid(input("Option type (call/put/both): "), ["call", "put", "both"])
    if optionType == 'both':
        LeftPutRightCall = getValid(input("Seperate puts for left tails and calls for right tails? (y/n): ")) == 'y'
    else:
        LeftPutRightCall = False
    riskFreeRate = getValid(input("Risk-free rate (e.g., 0.04 for 4%): "), type='float')
    graphType = getValid(input("Graph type (pdf/cdf/pmf/band): "), ["pdf", "cdf", "pmf", "band"]) # Band currently not implemented
    if graphType == 'pmf':
        pmfBins = 0
        while pmfBins <= 1:
            pmfBins = getValid(input("Number of bins for PMF (e.g., 10): "), type='int')
            if pmfBins <= 1:
                print("Please enter an integer greater than 1.")
    elif graphType == 'cdf':
        pmfBins = 0
    elif graphType == 'pdf':
        pmfBins = -1
    elif graphType == 'band':
        pmfBins = -2
    return lag, periods, optionType, riskFreeRate, pmfBins, LeftPutRightCall


def get_data(symbol, lag, periods, optionType): # Get + process option chain data
    tickerStr = symbol.upper()
    try:
        ticker = yf.Ticker(symbol)
    except Exception as e:
        print(f"Error retrieving data for {tickerStr}: {e}")
        return pd.DataFrame()
    try:
        tickerCurrentPrice = ticker.info['lastPrice']
    except: # if lastPrice not available, use last close price
        tickerCurrentPrice = ticker.history(period="1d")['Close'].iloc[0]
        print(tw.dedent("""Unable to retrieve lastPrice from info; using historical close price instead. Data may be inaccurate, especially for evening queries."""))
    allTickerDates = pd.to_datetime(ticker.options)
    if lag == 'day': # Filter expiration dates based on lag and periods
        expDates = allTickerDates[allTickerDates <= pd.Timestamp.today() + pd.offsets.BDay(periods)]
    elif lag == 'wk':
        expDates = allTickerDates[allTickerDates <= pd.Timestamp.today() + pd.DateOffset(weeks=periods)]
        expDates = expDates[expDates.weekday == 4] # Note potential gap if Friday is holiday
    elif lag == 'mon':
        current_month = pd.Timestamp.now().to_period('M')
        expDates = allTickerDates[allTickerDates <= pd.Timestamp.today() + pd.DateOffset(months=periods + 1)]
        expDates = expDates[expDates.to_period('M') != current_month]
        expDates = expDates.to_series().groupby(expDates.to_period('M')).min()
        expDates = expDates[:periods]
    fullChainData = pd.DataFrame()
    for date in expDates: # Iterate and concat through each expiration date (call, put, or both)
        date = date.strftime('%Y-%m-%d')
        opt = ticker.option_chain(date)
        callData = pd.DataFrame()
        putData = pd.DataFrame()
        if optionType in ['call', 'both']:
            callData = opt.calls.copy()
            callData['type'] = 'call'
        if optionType in ['put', 'both']:
            putData = opt.puts.copy()
            putData['type'] = 'put'
        chainData = pd.concat([callData, putData], ignore_index=True)
        chainData['expirationDate'] = date
        chainData = chainData[['strike', 'type','lastPrice', 'bid', 'ask', 'volume', 'expirationDate']]
        chainData['expirationDate'] = pd.to_datetime(chainData['expirationDate'])
        fullChainData = pd.concat([fullChainData, chainData], ignore_index=True)
    try: # "middlePrice" less noisy than lastPrice for options
        fullChainData['middlePrice'] = (fullChainData['bid'] + fullChainData['ask']) / 2
    except:
        fullChainData['middlePrice'] = fullChainData['lastPrice']
    fullChainData['spread'] = fullChainData['ask'] - fullChainData['bid'] # add spread
    fullChainData =  fullChainData[(fullChainData["spread"] >= 0) & 
                                   (fullChainData['strike'] > fullChainData['strike'].min()) &
                                   (fullChainData['strike'] < fullChainData['strike'].max())] # filter out invalid data
    return fullChainData

def graphProbabilityHeatmap(data, riskFreeRate, symbol, pmfBins, LeftPutRightCall, unimodal=True):
    ticker = yf.Ticker(symbol)
    try:
        tickerPriceLabel = ticker.info['lastPrice']
    except: 
        tickerPriceLabel = ticker.history(period="1d")['Close'].iloc[0]
    exp_ranges = data.groupby("expirationDate")["strike"].agg(["min","max"])
    K_min = exp_ranges["min"].max()
    K_max = exp_ranges["max"].min()
    K_grid = np.linspace(K_min, K_max, 400) # Standardized strike price grid
    cdfCallDict = {}
    pdfCallDict = {}
    cdfPutDict = {}
    pdfPutDict = {}
    pmfCallDict = {}
    pmfPutDict = {}
    cdfDict = {}
    pdfDict = {}
    pmfDict = {}
    binEdges = np.linspace(K_min, K_max, pmfBins + 1)
    QuantileRows = []
    for optionType in data['type'].unique(): # Separate by option type
        dataType = data[data['type'] == optionType]
        for expDate in dataType['expirationDate'].unique(): # Plot CDF for each expiration date
            dateSubset = dataType[dataType['expirationDate'] == expDate]
            expTime = pd.to_datetime(expDate).tz_localize("UTC")
            nowTime = pd.Timestamp.now(tz="UTC")
            dateSubset = dateSubset.sort_values("strike") # Sort by strike price
            dateSubset = dateSubset.drop_duplicates("strike", keep="last") 
            ## Get on-grid C(x) lowess -> spline -> d2C/dK2 -> PDF -> CDF
            T = (expTime - nowTime).total_seconds() / (365 * 24 * 3600)
            K = dateSubset['strike'].values  
            C = dateSubset['middlePrice'].values
            if optionType == 'put': # Convert put prices to call prices via put-call parity
                C = C - K * np.exp(-riskFreeRate * T) + tickerPriceLabel
            r = riskFreeRate
            C_interp_func = interp1d(K, C, kind='linear', bounds_error=False, fill_value='extrapolate')
            C_on_grid = C_interp_func(K_grid)
            s = len(K_grid) * np.var(C_on_grid) * 0.001
            # Fit spline on the standardized grid (prevents pdf getting cut off)
            CofK = sm.nonparametric.lowess(exog=K_grid, endog=C_on_grid, frac=0.2)
            CofK = UnivariateSpline(CofK[:,0], CofK[:,1], k=5, s=s)
            d2C = CofK.derivative(n=2)(K_grid)
            pdf = np.exp(r * T) * d2C
            pdf = np.maximum(pdf, 0)
            if unimodal: pdf = forceUnimodality(pdf, K_grid)
            area = np.trapezoid(pdf, K_grid)
            pdf = pdf / area
            cdf = np.cumsum((pdf[:-1] + pdf[1:]) / 2 * np.diff(K_grid))
            cdf = np.concatenate([[0], cdf])
            cdf = np.clip(cdf, 0, 1)
            pmf = np.zeros(max(pmfBins, 1))
            if pmfBins > 1: # Calculate PMF if specified
                for i in range(pmfBins):
                    mask = (K_grid >= binEdges[i]) & (K_grid < binEdges[i+1])
                    if mask.sum() > 1:
                        pmf[i] = np.trapezoid(pdf[mask], K_grid[mask])
                    elif mask.sum() == 1:
                        pmf[i] = pdf[mask][0] * (binEdges[i+1] - binEdges[i])
            if optionType == 'call':
                cdfCallDict[expDate] = cdf
                pdfCallDict[expDate] = pdf
                pmfCallDict[expDate] = pmf
            elif optionType == 'put':
                cdfPutDict[expDate] = cdf
                pdfPutDict[expDate] = pdf
                pmfPutDict[expDate] = pmf
            """
            pdf_plot = pdf * dateSubset['strike'].max() / (2 * pdf.max())
            plt.scatter(K, C, color='blue', label='Data points', zorder=2)
            plt.plot(K_grid, CofK(K_grid), color='red', label='Spline', linewidth=2)
            plt.plot(K_grid, pdf_plot, color='green', label='PDF', linewidth=1)
            plt.legend()
            plt.show()
            """
            q10, q25, q50, q75, q90 = [float(np.interp(p, cdf, K_grid)) for p in [0.10, 0.25, 0.50, 0.75, 0.90]]
    # Average call and put CDFs/PDFs if both present
    if len(data['type'].unique()) == 1:
        if 'call' in data['type'].unique():
            cdfDict = cdfCallDict
            pdfDict = pdfCallDict
            pmfDict = pmfCallDict
        elif 'put' in data['type'].unique():
            cdfDict = cdfPutDict
            pdfDict = pdfPutDict
            pmfDict = pmfPutDict
    else: 
        for expDate in pdfCallDict:
            if LeftPutRightCall: # CURRENTLY NOT WOKING -- FIX
                cdfDict[expDate] = np.where(K_grid < tickerPriceLabel, cdfPutDict[expDate], cdfCallDict[expDate])
                pdfDict[expDate] = np.where(K_grid < tickerPriceLabel, pdfPutDict[expDate], pdfCallDict[expDate])
                pmfDict[expDate] = np.where(binEdges[:-1] < tickerPriceLabel, pmfPutDict[expDate], pmfCallDict[expDate])
            else: 
                cdfDict[expDate] = (cdfCallDict[expDate] + cdfPutDict[expDate]) / 2
                pdfDict[expDate] = (pdfCallDict[expDate] + pdfPutDict[expDate]) / 2
                pmfDict[expDate] = (pmfCallDict[expDate] + pmfPutDict[expDate]) / 2
    # Plot graph
    K_labels = np.sort(data["strike"].unique())
    if pmfBins > 1:
        pmfDf = pd.DataFrame(pmfDict, index=binEdges[:-1]).T
        pmfDf.index = pd.to_datetime(pmfDf.index).strftime('%Y-%m-%d')
        pmfDf = pmfDf.iloc[:, ::-1]
        finalGraph = sns.heatmap(
            pmfDf.T, cmap='BuPu', vmin=0, vmax=round(pmfDf.values.max(),2))
        cbar = finalGraph.collections[0].colorbar
        cbar.ax.set_title('Probability', pad=10)
    elif pmfBins == 0:
        cdfDf = pd.DataFrame(cdfDict, index=K_grid).T
        cdfDf.index = pd.to_datetime(cdfDf.index).strftime('%Y-%m-%d')  
        finalGraph = sns.heatmap(
            cdfDf.T, cmap='coolwarm', vmin=0, vmax=1)
        cbar = finalGraph.collections[0].colorbar
        cbar.ax.set_title('Cumulative Probability', pad=10)
    elif pmfBins == -1:
        pdfDf = pd.DataFrame(pdfDict, index=K_grid).T
        pdfDf.index = pd.to_datetime(pdfDf.index).strftime('%Y-%m-%d')  
        finalGraph = sns.heatmap(
            pdfDf.T, cmap='coolwarm', vmin=0, vmax=pdfDf.values.max ())
        cbar = finalGraph.collections[0].colorbar
        cbar.ax.set_title('Probability', pad=10)
    elif pmfBins == -2:
        pass
    # Set labels, ticks, add current price line
    if pmfBins > 1:
        bin_centers = (binEdges[:-1] + binEdges[1:]) / 2  # length = pmfBins
        step = max(1, pmfBins // 10)
        ticks = list(range(0, pmfBins, step))
        ticklabels = [f"{bin_centers[::-1][i]:.1f}" for i in ticks]
        finalGraph.set_yticks(ticks)
        finalGraph.set_yticklabels(ticklabels)
    else: 
        finalGraph.invert_yaxis()
        label_positions = [np.argmin(np.abs(K_grid - k)) for k in K_labels]
        step = max(1, len(K_labels) // 10) 
        ticks = [label_positions[i] for i in range(0, len(K_labels), step)] + [
            np.interp(tickerPriceLabel, K_grid, np.arange(len(K_grid)))]
        ticklabels = [f'{K_labels[i]:.1f}' for i in range(0, len(K_labels), step)] + [f'{tickerPriceLabel:.2f}']
        finalGraph.set_yticks(ticks)
        finalGraph.set_yticklabels(ticklabels)
        finalGraph.get_yticklabels()[-1].set_fontweight('bold')
    plt.xticks(rotation=0)
    plt.xlabel('Expiration Date')
    plt.ylabel('Price ($)')
    plt.title(f'Option Probability Heatmap for {yf.Ticker(symbol.upper()).info["shortName"]} Based on Options Chain.')

def main():
    print(tw.dedent(""" 
        Welcome to the Option Probability Heatmap Generator.
        This tool uses yfinance options data to create probability heatmaps for security prices using an inverse Black-Scholes model.
        Note that the model does not account for dividends, market transaction costs, or early exercise of American options,
        and accuracy will vary across tickers and market conditions.\n"""))
    lag, periods, optionType, riskFreeRate, pmfBins, LeftPutRightCall = 'wk', 5, 'call', 0.04, 19, False
    periods = 5
    if getValid(input("Use default settings? (y/n): ")) == 'n':
        lag, periods, optionType, riskFreeRate, pmfBins, LeftPutRightCall = inputSettings()
    print()
    symbol = getValid(input("Enter ticker symbol (e.g., SPY): "), type='ticker')
    if symbol is None:
        return
    print()
    optionsData = get_data(symbol, lag, periods, optionType)
    graphProbabilityHeatmap(
        optionsData, riskFreeRate, symbol, pmfBins, LeftPutRightCall)
    plt.show()
    # print(optionsData[optionsData.groupby("expirationDate")["strike"].agg(["count","min","max"])])
    # optionsData.to_csv('option_prices.csv', index=False)
    # print(optionsData.head()) 

if __name__ == '__main__':
    main()