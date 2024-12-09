import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from alpha_vantage.timeseries import TimeSeries

# Step 1: Fetch historical stock data and earnings data
def fetch_earnings_data(stock_ticker):
    stock = yf.Ticker(stock_ticker)
    hist = stock.history(period="max")

    try:
        earnings = stock.earnings_dates
        if earnings is not None and not earnings.empty:
            earnings_dates = earnings.index.tolist()
        else:
            earnings_dates = []
            print(f"Earnings data for {stock_ticker} is unavailable or not retrieved properly.")
    except Exception as e:
        print(f"Error fetching earnings data for {stock_ticker}: {e}")
        earnings_dates = []

    return hist, earnings_dates

# Step 2: Fetch news articles using the NewsAPI
def fetch_news_for_stock(stock_ticker, api_key):
    url = f'https://newsapi.org/v2/everything?q={stock_ticker}&apiKey={api_key}'
    try:
        response = requests.get(url)
        articles = response.json().get('articles', [])
        headlines = [article['title'] for article in articles]
    except Exception as e:
        print(f"Error fetching news for {stock_ticker}: {e}")
        headlines = []
    return headlines

# Step 3: Sentiment analysis on news headlines
def analyze_sentiment(news_list):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [analyzer.polarity_scores(article)['compound'] for article in news_list]
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    return avg_sentiment

# Step 4: Fetch additional financial data from Alpha Vantage
def fetch_alpha_vantage_data(stock_ticker, api_key):
    try:
        ts = TimeSeries(key=api_key, output_format='pandas')
        data, _ = ts.get_daily(symbol=stock_ticker, outputsize='full')
    except Exception as e:
        print(f"Error fetching data from Alpha Vantage for {stock_ticker}: {e}")
        data = pd.DataFrame()
    return data

# Step 5: Calculate technical indicators (RSI and MACD)
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Step 6: Calculate the percentage move 1-week prior to earnings
def calculate_percentage_move(stock_data, earnings_dates, days_before=7):
    if stock_data.empty or not earnings_dates:
        print("Insufficient data for percentage move calculation.")
        return []
    stock_data['pct_change'] = stock_data['Close'].pct_change()
    features = []
    for date in earnings_dates:
        try:
            if date in stock_data.index:
                pct_change = stock_data['pct_change'].loc[date - pd.Timedelta(days=days_before):date].mean()
                features.append(pct_change)
        except Exception as e:
            print(f"Error processing date {date}: {e}")
    return features

# Remaining steps from the original code are unchanged but integrated cleanly
# ...

# Example usage
stock_ticker = "AAPL"
news_api_key = "YOUR_NEWS_API_KEY"
alpha_vantage_api_key = "YOUR_ALPHA_VANTAGE_API_KEY"

model = earnings_algorithm(stock_ticker, news_api_key, alpha_vantage_api_key)
