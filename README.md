# NASDAQ-Unveiled
Strategies for Predicting Closing Auctions  

By Joseph Bridges, Ali Kahn, Monica Liou, and Abhijit Anil  


## Abstract
This article summarizes our exploration in the Kaggle competition “Trading at the Close” hosted by Optiver between September 20th and December 20th, 2023 as part of the course Advanced Machine Learning (University of Texas at Austin, MSBA). This competition focuses on predicting the short term price movements in the closing auction for NASDAQ companies.

As part of this project, we performed an exploratory data analysis (EDA), researched various data preprocessing techniques and challenges, and employed four different machine learning methods to attempt to predict the data. We found XGBoost to be the most successful both in predictive power

## Introduction
### The Problem
Every business day, markets host an ‘auction’ in the last 10 minutes of trading (usually, from 3:50–4:00 pm EST). Market participants submit orders into the auction consisting of ‘bids’–a price that the market participant would like to buy a stock at–and ‘asks’–a price that a stockholder would wish to sell a stock at. Naturally, ask prices tend to be higher than bid prices, as those willing to sell the stock want to get a higher price for their sale then those wishing to buy. This means that matching ask and bids (which corresponds to a completed trade) is an optimization problem: finding the final closing price (the market must decide on a single price by the end of the auction) that maximizes the number of trades which occur. Such a price is called the ‘uncross’ price, and market participants can attempt to predict the uncross price at close.

The problem of optimizing the cross-price is baked into the larger problem of predicting the short term movements of the uncross price throughout the auction. As orders come in during the 10 minutes the auction is open, market makers will continually work on the problem of finding the best auction price in order to decide if/how they should enter the auction.

### Data Description
The data provided by Kaggle is historical data of the daily closing auction order books for over 200 different stocks. The rows of the data give a glimpse of the current values in the book for a given time. Columns include current number of bids, current number of asks, the current reference price (the weight average uncross price put into basis points), the bid-ask imbalance (the number of unmatched orders at the reference price), as well as other statistics which characterize these core figures. There are 5 million rows in the Kaggle data and these rows are uniquely indexed by a combination of date, stock ID, and time (bucketed by the second). The goal is to predict the the target variable, the reference price 6 rows (60 seconds) after the current time.
