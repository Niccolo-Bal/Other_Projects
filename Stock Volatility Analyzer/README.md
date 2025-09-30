## Stock Volatitlity Analyzer
This is a UI which allows the user to analyze the volatiltiy of stocks either through specific stocks or top N stocks by market cap.
It contains two python scripts, both of which serve the same purpose, one of which uses cache to eliminate a lot of delay that comes with the yf API pulls (named "optimized").
There are some pre-cached ticker data, but the UI has the ability to both clear this cache and download more batches.
