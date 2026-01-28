# Pairs Trading Testing + POC
These files are designed to guide potential pairs trading opportunities.

Candidate_Test_GUI.py is a simple UI which gives key data on the cointegration between two tickers (from yfinance), with basic cointegration statistics such as: residual ADF test, EG test, correlation/fit, and regression, as well as mathplotlib graphs of the spread between the assets, as well as the assets themselves.

Simple_Backtest.py calculates the theoretical profit that would result from spread trading on set assets during a given period (using β_0, β_1 from trailing 8mo, entering at |z| = 1.96 and exiting at |z| = 0.5)

These tests could advise on potential pairs trades, but take the information they provide (especially the backtest) at face value.
