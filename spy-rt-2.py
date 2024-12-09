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
    barset = barset[barset['symbol'] == symbol] if 'symbol' in barset else barset  # Check for 'symbol' column
    return barset

def get_latest_price(symbol):
    """
    Fetch the latest price for the given symbol using Alpaca's latest trade endpoint.
    """
    latest_trade = api.get_latest_trade(symbol)
 #   print(f"Debug: Latest Trade Response: {latest_trade}")
    return latest_trade.price




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
    print(f"Starting real-time analysis for {SYMBOL}...")

    while True:
        now = datetime.datetime.now()
        if now.time() >= END_TIME:
            print("Market closed. Exiting script.")
            break
        if now.time() < START_TIME:
            print("Waiting for market to open...")
            time.sleep(60)
            continue

        try:
            # Fetch the latest price
            latest_price = get_latest_price(SYMBOL)
            print(f"Current Price: {latest_price:.2f}")

            # Fetch historical data and append the latest price
            data = get_historical_data(SYMBOL)
            if len(data) < SUPPORT_RESISTANCE_WINDOW:
                print("Insufficient data for analysis. Waiting...")
                time.sleep(CHECK_INTERVAL)
                continue

            # Add the latest price as the most recent row in historical data
            new_row = {'close': latest_price}  # Add other required columns if needed
            data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

            # Recalculate RSI and support/resistance
            data['RSI'] = calculate_rsi(data)
            support, resistance = calculate_support_resistance(data)

            current_rsi = data['RSI'].iloc[-1]
            print(f"RSI: {current_rsi:.2f} | Support: {support:.2f} | Resistance: {resistance:.2f}")

            # Signal logic
            if current_rsi < RSI_OVERSOLD:
                print(f"Signal: BUY CALL at {latest_price:.2f}")
            elif current_rsi > RSI_OVERBOUGHT:
                print(f"Signal: BUY PUT at {latest_price:.2f}")
            elif latest_price < support:
                print(f"Signal: SELL CALL (Support Broken) at {latest_price:.2f}")
            elif latest_price > resistance:
                print(f"Signal: SELL PUT (Resistance Broken) at {latest_price:.2f}")

        except Exception as e:
            print(f"Error fetching or analyzing data: {e}")

        time.sleep(CHECK_INTERVAL)





# Main Function
def main():
    analyze_market()

if __name__ == "__main__":
    main()
