
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

    # Fetch earnings dates and handle cases where the data might be missing
    try:
        earnings = stock.earnings_dates
        if earnings is not None and not earnings.empty:
            earnings_dates = earnings.index.tolist()
        else:
            earnings_dates = []
            print(f"Earnings data for {stock_ticker} is unavailable or not retrieved properly.")
    except AttributeError:
        earnings_dates = []
        print(f"Earnings data for {stock_ticker} could not be fetched.")

    return hist, earnings_dates

# Step 2: Fetch news articles using the NewsAPI
def fetch_news_for_stock(stock_ticker, api_key):
    url = f'https://newsapi.org/v2/everything?q={stock_ticker}&apiKey={api_key}'
    response = requests.get(url)
    articles = response.json().get('articles', [])
    headlines = [article['title'] for article in articles]
    return headlines

# Step 3: Sentiment analysis on news headlines
def analyze_sentiment(news_list):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [analyzer.polarity_scores(article)['compound'] for article in news_list]
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    return avg_sentiment

# Step 4: Fetch additional financial data from Alpha Vantage
def fetch_alpha_vantage_data(stock_ticker, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily(symbol=stock_ticker, outputsize='full')  # Daily data
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
    if not earnings_dates:
        print("No earnings dates available to calculate percentage moves.")
        return []

    stock_data['pct_change'] = stock_data['Close'].pct_change()
    features = []

    for date in earnings_dates:
        if date in stock_data.index:
            pct_change = stock_data['pct_change'].loc[date - pd.Timedelta(days=days_before):date].mean()
            features.append(pct_change)

    return features

# Step 7: Label earnings success/failure
def label_earnings_success(earnings_data, stock_data):
    try:
        # Assuming some processing occurs here
        labels = []  # Initialize labels to ensure it's defined

        # Add the logic for labeling earnings success/failure
        for data_point in earnings_data:
            # Your logic for creating the labels based on earnings data and stock data
            labels.append(1 if data_point["earnings_success"] else 0)

        print(f"Labels: {labels[:10]}")  # Print the first 10 labels for inspection
        return labels

    except Exception as e:
        print(f"Error in labeling earnings success: {e}")
        return []  # Return an empty list or handle the error appropriately


# Step 8: Standardize features and combine them for training
def prepare_features_for_model(stock_data, sentiment_score, percentage_moves, rsi, macd):
    print("Preparing features for the model...")
    print(f"Sentiment score: {sentiment_score}")
    print(f"Percentage moves: {percentage_moves}")
    print(f"RSI: {rsi}")
    print(f"MACD: {macd}")

    # Ensure sentiment_score is broadcasted correctly (to match the length of other features)
    sentiment_feature = np.full((len(stock_data), 1), sentiment_score)

    # Handle other feature lengths, ensuring they align with stock_data length
    if len(percentage_moves) != len(stock_data):
        print(f"Adjusting percentage moves from {len(percentage_moves)} to match stock data length: {len(stock_data)}")
        percentage_moves = np.resize(percentage_moves, len(stock_data))
    
    if len(rsi) != len(stock_data):
        print(f"Adjusting RSI from {len(rsi)} to match stock data length: {len(stock_data)}")
        rsi = np.resize(rsi, len(stock_data))

    if len(macd[0]) != len(stock_data):
        print(f"Adjusting MACD from {len(macd[0])} to match stock data length: {len(stock_data)}")
        macd = (np.resize(macd[0], len(stock_data)), np.resize(macd[1], len(stock_data)))

    # Now stack them into a single array
    features_list = [sentiment_feature, percentage_moves, rsi, macd[0], macd[1]]
    features = np.column_stack(features_list)

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled


# Step 9: Create and train the XGBoost model
def train_model(features, labels):
    print(f"Features length: {len(features)}")
    print(f"Labels length: {len(labels)}")

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    xgb_model = XGBClassifier(random_state=42)
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 10]
    }
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model

# Step 10: Main function
def earnings_algorithm(stock_ticker, news_api_key, alpha_vantage_api_key):
    stock_data, earnings_data = fetch_earnings_data(stock_ticker)
    news_headlines = fetch_news_for_stock(stock_ticker, news_api_key)
    sentiment_score = analyze_sentiment(news_headlines)
    alpha_vantage_data = fetch_alpha_vantage_data(stock_ticker, alpha_vantage_api_key)
    percentage_moves = calculate_percentage_move(stock_data, earnings_data)
    rsi = calculate_rsi(stock_data)
    macd = calculate_macd(stock_data)
    labels = label_earnings_success(earnings_data, stock_data)
    features = prepare_features_for_model(stock_data, sentiment_score, percentage_moves, rsi, macd)
    
    # Log the shapes of features and labels for debugging
    print(f"Features shape: {features.shape if features is not None else 'None'}")
    print(f"Labels shape: {len(labels) if labels is not None else 'None'}")
    
    if features is None or labels is None:
        print("Insufficient data to train the model.")
        return None
    model = train_model(features, labels)
    return model


# Example usage
stock_ticker = "AAPL"
news_api_key = "YOUR_NEWS_API_KEY"
alpha_vantage_api_key = "YOUR_ALPHA_VANTAGE_API_KEY"
model = earnings_algorithm(stock_ticker, news_api_key, alpha_vantage_api_key)
