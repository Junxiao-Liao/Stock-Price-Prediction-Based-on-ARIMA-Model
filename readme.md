# Stock Price Prediction Based on ARIMA Model

## 1. Problem Description

Time series prediction methods are commonly used prediction techniques. They have a complete theoretical foundation. With people's enthusiasm for the stock market, many scholars have also tried to apply time series prediction to stock price prediction. Specifically, predicting stock prices with time series is to treat stock prices or price indices as time series that change over time, and to predict future trends and patterns through establishing reasonable time series models. This paper selects the closing price of Chang'an Automobile stocks as empirical analysis data for time series, and predicts the patterns and trends of stock price changes in the GEM market by establishing an ARIMA model. It should be noted that the market conditions are ever changing, and the time series analysis method only hopes to obtain useful information from the historical data to predict future trends, without considering the reasons affecting stock price changes, so it usually only provides an intuitive analysis for short-term prediction.[1] 

## 2. Data Crawling

### 2.1 Crawling Process

First, the selenium library is used, whose advantage is that it can simulate the browser usage well, and can be used to simply write crawler programs without being easily detected; second, the time library is used to pause for a period of time to mimic the real behavior of the browser while waiting for the page to load; finally, the re library is used to analyze the urls of webpages.[2]

### 2.2 Analyze urls 

When crawling historical data of stocks on NetEase Finance, it is observed that the storage address of stock information is in the format of `http://quotes.money.163.com/trade/lsjysj_xxxxxx.html`, where xxxxxx is the stock code. Clicking the "Download Data" button on this address and then clicking the "Download" button can download all historical data of the stock.

### 2.3 Get data

Since only the historical data of stock 000625 needs to be obtained this time, the storage address can be uniquely determined. The key code is as follows:

```python
url = 'http://quotes.money.163.com/trade/lsjysj_000625.html#01b07'
browser.get(url)
time.sleep(2)
browser.find_element(
    By.XPATH, '//*[@id="downloadData"]').click() # press button
time.sleep(2) 
browser.find_element(
    By.XPATH, '/html/body/div[2]/div[5]/div[2]/form/div[3]/a[1]').click()
time.sleep(2)
```

## 3. Data Preprocessing 

### 3.1 Import Data

Directly use the read_csv method of the pandas library, the code is as follows:

```python
df = pd.read_csv('000625_source.csv', sep=',')
```

### 3.2 Missing Value Check

After checking, the data is complete without missing values. The code is as follows:

```python
na_cols = df.isnull().any(axis=0)
```

## 4. Time Series Preprocessing

If too little data is selected, it will be impossible to fully extract information from the historical data. Selecting too much data will cause unnecessary errors because stock prices with longer intervals will have less impact on predicting future stock prices. Therefore, the most recent 150 days of data are used for analysis. For convenience, only the closing price of each day is taken as the stock price of the day.

The key code is as follows:

```python
data = data.head(150)
dataUtil = data.iloc[:, ‘Date’]
```

## 5. ARIMA Model Preprocessing

### 5.1 Time Series Chart Inspection

First, use the intuitive time series chart to observe whether the time series is stationary. The result is as shown in the figure below:

Obviously it is not a stationary time series and smoothing processing is needed. 

### 5.2 Differencing for Stationarity

Perform first order differencing on the original data. The related values are shown in Table 1. After processing, it can be seen that the absolute value of the ADF statistic dy is 3.9615702003502933, greater than 3.8746401679188405, which is the significance level when the significance level is 0.05. Therefore, the hypothesis that there is a single root is accepted, indicating that the first order differenced sequence is stationary, so d=1.

**Table 1: ADF value significance comparison after first order differencing**

| Category | adf value |
|-|-|  
| Result | -3.9615702003502933 |
| 1% | -4.454630091433345 |
| 5% | -3.8746401679188405 |
| 10% | -3.579316526412991 |

### 5.3 Autocorrelation and Partial Autocorrelation Plots  

The autocorrelation and partial autocorrelation plots are shown in figures 2 and 3:

It can be observed that the autocorrelation graph has a truncated tail and the partial autocorrelation graph has a trailing tail. According to theory, the MA model should be used, which is included in the subsequent grid search.

### 5.4 White Noise Test

The white noise test values are shown in Table 2:

**Table 2:** White noise test values

| lb_stat | lb_pvalue |
|-|-|
| 3.485312 | 0.061916 |
| 3.485558 | 0.175033 |
| 5.420166 | 0.143492 |
| 8.508024 | 0.074644 |
| 9.200742 | 0.101320 |
| 10.421292 | 0.107995 |
| 13.678446 | 0.057205 |
| 13.909369 | 0.084158 |

When lag=1,4,7, the p-value is close to 5%, and it is still treated as a pass. It is considered that the data is a non-white noise sequence with correlation and certain rules to follow.

## 6. Model Building and Forecasting

### 6.1 Splitting Training and Test Sets

After differencing, there are still 149 data remaining. The first 144 are used as the training set and the last 5 as the test set.

### 6.2 Grid Search for Optimal Parameters 

For the ARIMA library function ARIMA(train, order(p, d, q)), grid search is used to find the best matching order value. The functions evaluate_arima_model and evaluate_models are defined, with key code as follows:

```python
def evaluate_arima_model(X, arima_order):
    size = len(X) - 5
    train_tmp, test_tmp = X[:size], X[size:]
    tmp_model = ARIMA(train_tmp, order=arima_order).fit()
    tmp_pred = tmp_model.forecast(5)
    error = mean_squared_error(tmp_pred, test_tmp)
    return error

def evaluate_models(dataset, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
```

Through the search, the best parameters are found to be ARIMA(8, 1, 6), with the minimum MSE of 0.102.

### 6.3 Model Forecasting

The forecast results are shown in Table 3:

**Table 3:** Model forecast results

| Forecast Value | Actual Value |
|-|-|
| 0.117384 | 0.45 |
| -0.259985 | 0.54 |  
| -0.035829 | 1.76 |
| 0.093502 | -0.17 |
| -0.139943 | -1.36 |

The predicted values and actual values are plotted on the same coordinate system, as shown in the figure below:

## 7. Result Analysis 

### 7.1 Intuitive Analysis from Figure 4

The results are almost the same as the actual values in terms of increasing and decreasing trends, but the numerical values are not yet accurate enough.

### 7.2 Residual Analysis

As shown in the figure, the residual distribution has good normality and the model fitting is relatively good.

## References

[1] Wu Yuxia & Wen Xin. (2016). Short-term stock price prediction based on ARIMA model. Statistics and Decision (23), 83-86. doi:10.13546/j.cnki.tjyjc.2016.23.051. 

[2] Zhihu user: Heimu Bird. How to use crawlers to grab stock market data and generate analysis reports [EB/OL]. https://www.finlab.tw/%E7%94%A8%E7%88%AC%E8%9F%B2%E7%88%AC%E5%85%A8%E4%B8%96%E7%95%8C%E8%82%A1%E5%83%B9/, 2017–02–22/2018–05–31.
