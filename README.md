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

### Potential Models
The problem of predicting the close price is a time series problem, and so we expect to find the best results with models that work well with time series data. Cursory research on this topic suggests that XGBoost as well as various Neural Net architectures provide the most competitive outcomes for time series problems involving financial data.

## EDA
![image](https://github.com/MonicaLiou1025/NASDAQ-Unveiled/assets/140920765/f2d19390-4ae2-4a48-b394-1ac579cd1881)
In our EDA of stock_id = 0, we observed distinct periods where bid, ask, and WAP lines closely tracked together, indicating lower market fluctuations and heightened market stability. Conversely, there were intervals with substantial spreads between bid and ask prices, suggesting increased volatility, potentially in response to market news or events. This analysis underscores the dynamic nature of financial markets, where moments of stability and turbulence coexist, offering valuable insights for researchers and investors alike.

![image](https://github.com/MonicaLiou1025/NASDAQ-Unveiled/assets/140920765/a205b8d2-5055-46b2-ad19-698fa141ff5a)
The chart presents a time series analysis of two data sets: the Weighted Average Price (WAP) in blue and a target variable in green, likely representing a financial metric. There appears to be an inverse relationship between the two; as the target variable increases, the WAP often decreases, suggesting a potential predictive or contrarian relationship. This could imply that the target variable may be useful for forecasting or understanding future movements in the WAP, but further statistical analysis would be required to validate this correlation and its potential for predictive modeling.

![image](https://github.com/MonicaLiou1025/NASDAQ-Unveiled/assets/140920765/1b491335-c1e1-44bc-b7b4-2ab7ae1004b3)
The displayed chart illustrates the interactions between the WAP and spot prices within a trading day, highlighting the volatility of near prices and less frequent but significant movements of far prices. Key observations include the divergence around the 400-time mark, which may signal a disruptive market event or arbitrage opportunity, and the overall stability of the WAP, suggesting it is less susceptible to short-lived market fluctuations. Traders might leverage these patterns to identify potential trading strategies, although caution should be exercised to corroborate these insights with further market data.

## Data Preprocessing & Challenges
The Trading at the Close dataset is an example of time series data. Time series datasets present unique problems unlike any of those in other machine learning schemas. This is because future time series data points are unknown, and occur sequentially after past data points. Test data sets need to be fed to time series models in the order in which they are indexed, as future data points can contain information about data points in the past. If an online learning model is continuously adapting to future data points, then it may be able to predict past data points with dubiously high accuracy.

### Challenge I: Time Gap
Data pre-processing techniques popular for time series data include lag features (columns that contain information from previous rows in the data), moving averages, and other time-based features that target non-linear patterns in the data (see the conclusion for a discussion of Markov Chains).

The Trading at the Close dataset provides a unique challenge in that rather than simply predicting movements in the next time frame, the competition asks us to predict six time frame ahead (one minute into the future, or six rows of data). In a real application, we would be predicting up to ten minutes ahead to when the market closes. This time gap weakens the utility of lag features, as they indicate the interaction of features over shorter time frames than we would like.

### Challenge II: Martingale Problem
Another thing to consider with the Trading at the Close dataset is that the data is financial in nature. Financial time series data are often assumed to be ‘Martingales’, meaning that future price movements are assumed to be independent of past movements. This is a strong assumption and needs to be relaxed if there is to be any hope for predictive models that do better than the baseline. If it is true, then any given model should not do better than a baseline model that predicts future data points according to random movements.

We know that the Martingale assumption is violated, at least in part, for the Trading at the Close data set. According to Optiver’s introductory notebook [3], a simple mapping that beats the baseline incorporates the basis point change in reference price implied by the current buy/sell imbalance. That is, at a given time we have the reference price (wap for the current order book) and the buy/sell imbalance that has accumulated in that time bucket can be solved for a change in the reference price. If we predict just this change in the reference price (as the simple mapping does), and remain agnostic about future movements in the next 6 timeframes, then we can beat the baseline even under the strong Martingale assumption.

### Data Preprocessing Decisions
Given these challenges, we decided to stick to simple moving averages in feature engineering. Additionally, we standardized the data according to each stock ID. That is, rather than standardize a whole column, we thought it would be best to consider each stock to be its own data set and group by stock ID before standardizing the data (so that each feature had mean zero and variance one).

## XGBoost
We explored an XGBoost model for its time series prowess[2] and as a benchmark our data preprocessing techniques. An un-tuned XGBoost model with no additional features results in reasonably competitive performance (in relation to the relative error size, not placing!) with other models submitted to Kaggle. Because of this, we decided to test our data preprocessing techniques with respect to XGBoost rather than other models since it was relatively quick to train.

We found the standardization of columns by stock ID to slightly improve MAE, perhaps because “a model trained without scaling the adjusted closing prices will only output predictions around the range of the prices in the train set” [1]. Additionally, the moving average features we added in adata preprocessing seemed to neither help nor hurt the model.

## ARIMA Model
In addressing our time-series challenge, we sought to evaluate the effectiveness of ARIMA on our dataset. Before delving into the analysis, it’s essential to establish a comprehensive understanding of the model.

### Understanding ARIMA:
ARIMA, an acronym for Auto-Regressive, Integrated Moving Average, is comprised of three key components: Auto-Regressive, Integrated, and Moving Average.

Auto-Regressive (AR): Determines the present value based on historical values.
Integrated (I): Integrates the AR and MA elements by computing differences between current and past values.
Moving Average (MA): Smoothens variations to reveal the underlying trend.

The inclusion of the Integrated component serves a crucial purpose. Relying solely on AR and MA proves insufficient. These components operate effectively only on data exhibiting a stationary trend. In cases where datasets exhibit upward or downward trends, the Integrated component becomes key. Its primary function is to transform the dataset into a stationary trend, achieved through the computation of differences between current and past values. This adjustment ensures the applicability of AR and MA components to a broader spectrum of datasets, allowing for more accurate and comprehensive modeling.

### Parameters p, d, and q:
p: Represents the number of past values considered for the AR component.
q: Denotes the number of moving averages applied.
d: Specifies the number of past values subject to differentiation.

### ARIMA Model Training:
The ARIMA model was trained using Date IDs ranging from 0 to 477 for training purposes and 478 to 480 for testing. With a dataset encompassing 200 Stock IDs, we systematically tested various values of p (ranging from 0 to 3), d (ranging from 0 to 2), and q (ranging from 0 to 7). The objective was to identify the order yielding the lowest Mean Absolute Error (MAE) for each Stock ID in the test data.

Acknowledging the limitations of ARIMA, we sought a more intricate solution and redirected our focus towards Neural Networks.

## Neural Net Model

### Feed Forward / MLP Neural Network
The neural network architecture we’ve implemented is a combination of both categorical and numerical features, often referred to as a wide and deep neural network. This architecture was introduced in the paper “Wide & Deep Learning for Recommender Systems” by Google researchers. The wide part captures memorization of feature interactions, while the deep part captures generalization to new, unseen feature combinations.

### In our specific model:
The embedding layers for categorical variables create the deep part of the network.

The dense layers for numerical variables and subsequent dense layers contribute to both the deep and wide aspects of the network.

This architecture is particularly useful in scenarios where we have both low-level features that benefit from memorization and high-level features that benefit from generalization.

To summarize, it’s a wide and deep neural network architecture that combines the strengths of memorization and generalization for improved predictive performance.
![image](https://github.com/MonicaLiou1025/NASDAQ-Unveiled/assets/140920765/29931b3b-e3c2-4486-820a-a35cfe47df2e)



## Long Short Term Memory — A type of Recurrent Neural Network
The model we’ve created is a hybrid neural network that combines both categorical features and sequential (time series) data. It can be categorized as a combination of a Wide & Deep architecture and a Sequence-to-Sequence (Seq2Seq) architecture. Let me break down the characteristics:

### Wide & Deep:
The embedding layers for categorical features, followed by a dense layer, form the “wide” part. This part helps the model learn explicit relationships and memorize specific combinations of categorical features.

The LSTM layers processing sequential inputs form the “deep” part. This part captures temporal dependencies and learns patterns in time series data.

### Sequence-to-Sequence (Seq2Seq):
The multiple LSTM layers processing different time series inputs indicate a sequence-to-sequence architecture. Each LSTM processes a sequence of data and summarizes it into a fixed-size representation.

The concatenation of these representations and subsequent dense layers contribute to the final output.

### Overall:
The model is a hybrid that leverages both wide and deep architectures to capture both memorization of specific categorical feature combinations and generalization of sequential patterns over time.

In summary, it’s a Wide & Deep Seq2Seq neural network. This type of architecture is commonly used in scenarios where both categorical and sequential information are crucial for making accurate predictions, such as in time series forecasting or sequence prediction tasks with additional categorical features.

![image](https://github.com/MonicaLiou1025/NASDAQ-Unveiled/assets/140920765/b2db4164-c964-47bb-a7db-4b1576fce6da)


## Results
ARIMA Model Performance:
The MAE values spanned from 2.5422 (Stock ID: 4) to 16.6138 (Stock ID: 31); these figures were greater than those achieved with XGBoost. ARIMA’s limitation lies in its exclusive focus on historical values and moving averages, lacking the adaptability inherent in XGBoost, which considers various influencing factors.

### Feed Forward Neural Net
This model achieved a test MAE of 6.4508 and found optimal results with 10 epochs and a SGD batch size of 1024 (2¹⁰). It took considerably less time to train than the LSTM model, but considerably more than XGBoost.

### LSTM
This model achieved a test MAE of 6.4519 and found optimal results with the same 10 epochs and batch size of 1024. The more complicated architecture of LSTM led to noticeably longer training time than any of our other models.

### XGBoost
This model achieved a competitive results with the Neural Nets, yielding a 4.598 MAE and a substantially shortened training time.

### Overall
XGBoost seems to have effectively incorporated what little information there is about the target variable contained in the data set. At the time of this post, the current 1st place model has an MAE of 5.3070, just ~4% below the MAE achieved by our XGBoost model.

## Conclusion
Financial time series data presents a variety of challenges that data scientists must hope to overcome if they are to predict future trends. We have found that in the Trading at the Close dataset, there is little information to be gleaned from the data with respect to predicting the target variable.

That being said, in a competitive environment such as a stock market, small edges in predictive power can yield massive results, as having just a little bit more information than your competitors offers a huge potential for making successful trades. XGBoost and Neural Nets such as LSTM and simple MLPs can incorporate the various bits of information in the non-linearities of the data and provide competitive results.

Future improvements on these models will likely come from careful and clever data preprocessing techniques and feature engineering. Hidden interaction terms can reveal more about the data than our models may be able to precisely uncover. For example, interesting applications of Markov Chains have found their way into financial forcasting. For a similar problem that takes place in continuous trading time (i.e. before the closing auction), Markov chains could be used to create transition probabilities between the states of a stock [4].

In the real world, we would want to bring in additional information about each of these stocks such as sentiment score of news, financial history, and growth prospects so that we could hope to generate better predictions about how they might move in the future.

## References
[1] Ng, Yibin. “Forecasting Stock Prices using XGBoost (Part 1/5)” October 26, 2019. Medium. link  
[2] Jian, Zhenah. “An Attention GRU-XGBoost Model for Stock Market Prediction Strategies” November 27, 2022. 2022 4th International Conference on Advanced Information Science and System (AISS 2022). link  
[3] Forbes, Tom. “Optiver — Trading At The Close Introduction” September 20, 2023. Kaggle link  
[4] Amunategui, Manuel. “Stock Market Predictions with Markov Chains and Python” March 26, 2019. Viral ML Blog link  
[5] Cheng, Heng-Tze, et al. “Wide & deep learning for recommender systems.” June 26, 2016. Proceedings of the 1st workshop on deep learning for recommender systems. link  
