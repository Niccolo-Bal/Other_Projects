# Option Implied Price Probability Heatmap

This project uses Yahoo Finance options chain data to generate the implied probability of the stock being a particular price, by taking the second derivative of the call price as a funciton of strike price. 

option_prob_heatmap visualizes this in a sns heatmap over a certain time period, while CofKVisualization graphs C(K), the spline it generates, and the pdf for a specific date.

As of 1/13, there is no "band" functionality for heatmap generation, and put call conversions can cause buggyness (by default uses call-only).