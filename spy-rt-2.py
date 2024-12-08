import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import talib
import datetime
import time

# Alpaca API Credentials
API_KEY = 'PKKH4UZ675SAL8SAKUVE'
SECRET_KEY = 'bQyBNPYd1aLIOabsh3McUcMpLDzGEx9mBZyxjaVq'
BASE_URL = 'https://paper-api.alpaca.markets/v2'  # Replace with live URL if needed

# Connect to Alpaca
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# Parameters
SYMBOL = 'SPY'
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
SUPPORT_RESISTANCE_WINDOW = 30
CHECK_INTERVAL = 60  # Check every 60 seconds
START_TIME = datetime.time(9, 30)  # Market open
END_TIME = datetime.time(16, 0)  # Market close

# Function to fetch real-time data
def get_historical_data(symbol, interval="1Min", limit=100):
    """
    Fetch historical minute data from Alpaca for analysis.
    """
    barset = api.get_bars(symbol, interval, limit=limit).df
    barset = barset[barset['symbol'] == symbol]  # Filter only for the specific symbol
    return barset

# Function to calculate RSI
def calculate_rsi(data, period=RSI_PERIOD):
    """
    Calculate RSI using TA-Lib.
    """
    return talib.RSI(data['close'].values, timeperiod=period)

# Function to calculate dynamic support/resistance
def calculate_support_resistance(data, window=SUPPORT_RESISTANCE_WINDOW):
    """
    Calculate the rolling high (resistance) and low (support).
    """
    high = data['close'].rolling(window=window).max().iloc[-1]
    low = data['close'].rolling(window=window).min().iloc[-1]
    return low, high

# Real-time analysis loop
def analyze_market():
    """
    Analyze the market in real-time and print buy/sell signals.
    """
    print(f"Starting real-time analysis for {SYMBOL}...")
    while True:
        # Get current time
        now = datetime.datetime.now()
        if now.time() >= END_TIME:
            print("Market closed. Exiting script.")
            break
        if now.time() < START_TIME:
            print("Waiting for market to open...")
            time.sleep(60)  # Check again in 60 seconds
            continue

        try:
            # Fetch latest data
            data = get_historical_data(SYMBOL)
            if len(data) < SUPPORT_RESISTANCE_WINDOW:
                print("Insufficient data for analysis. Waiting...")
                time.sleep(CHECK_INTERVAL)
                continue

            # Calculate RSI and support/resistance
            data['RSI'] = calculate_rsi(data)
            support, resistance = calculate_support_resistance(data)

            # Get the latest values
            current_price = data['close'].iloc[-1]
            current_rsi = data['RSI'].iloc[-1]
            print(f"Current Price: {current_price:.2f} | RSI: {current_rsi:.2f} | Support: {support:.2f} | Resistance: {resistance:.2f}")

            # Signal logic
            if current_rsi < RSI_OVERSOLD:
                print(f"Signal: BUY CALL at {current_price:.2f}")
            elif current_rsi > RSI_OVERBOUGHT:
                print(f"Signal: BUY PUT at {current_price:.2f}")
            elif current_price < support:
                print(f"Signal: SELL CALL (Support Broken) at {current_price:.2f}")
            elif current_price > resistance:
                print(f"Signal: SELL PUT (Resistance Broken) at {current_price:.2f}")

        except Exception as e:
            print(f"Error fetching or analyzing data: {e}")

        # Wait before the next check
        time.sleep(CHECK_INTERVAL)

# Main Function
def main():
    analyze_market()

if __name__ == "__main__":
    main()
