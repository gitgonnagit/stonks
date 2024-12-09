import yfinance as yf 
import pandas as pd
import numpy as np
import talib
import datetime
import time
from alpaca_trade_api.rest import REST, TimeFrame

# Alpaca API keys
API_KEY = 'PKKH4UZ675SAL8SAKUVE'
API_SECRET = 'bQyBNPYd1aLIOabsh3McUcMpLDzGEx9mBZyxjaVq'
BASE_URL = 'https://paper-api.alpaca.markets/v2'  # Use live API URL for live trading

# Initialize Alpaca API
alpaca = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def is_market_open():
    current_time = datetime.datetime.now()
    market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= current_time <= market_close

def download_data(symbol, timeframe=TimeFrame.Minute):
    # Fetch the data from Alpaca (or your source)
    df = alpaca.get_bars(symbol, timeframe, limit=100).df
    if 'close' in df.columns:
        df = df[['close']]
    else:
        print("Warning: 'close' column not found.")
        df = df[['close_price']]  # Adjust based on inspection
    return df

def calculate_rsi(data, period=14):
    return talib.RSI(data['close'].values.ravel(), timeperiod=period)

def calculate_support_resistance(data, window=30):
    """
    Calculate dynamic support and resistance based on recent high/low
    """
    high = data['close'].rolling(window=window).max().iloc[-1]  # Max high in window
    low = data['close'].rolling(window=window).min().iloc[-1]  # Min low in window
    return low, high

def choose_strike(symbol, price):
    # Fetch the options data for the symbol (e.g., SPY)
    options_data = yf.Ticker(symbol).options
    atm_strike = round(price)  # Round to nearest strike price
    next_strike = atm_strike + 1  # Strike 1 above the ATM strike
    return next_strike

def real_time_strategy():
    capital = 10000  # Starting capital
    position = 0  # 0 for no position, 1 for call, -1 for put
    entry_price = 0
    entry_time = None
    profit_loss = 0
    options_position = None  # To track the options bought (strike, type, qty)

    print(f"Strategy started at {datetime.datetime.now()}")

    while True:
        data_1m = download_data('SPY', TimeFrame.Minute)
        data_5m = download_data('SPY', TimeFrame.Minute * 5)
        data_10m = download_data('SPY', TimeFrame.Minute * 10)
        data_15m = download_data('SPY', TimeFrame.Minute * 15)

        rsi_1m = calculate_rsi(data_1m, 14)[-1]
        rsi_5m = calculate_rsi(data_5m, 14)[-1]
        rsi_10m = calculate_rsi(data_10m, 14)[-1]
        rsi_15m = calculate_rsi(data_15m, 14)[-1]
        
        close_price = data_1m['close'].iloc[-1]
        print(f"SPY Current Price: {close_price}, RSI 1m: {rsi_1m}")

        # Dynamic support and resistance calculation
        support, resistance = calculate_support_resistance(data_1m)
        print(f"Dynamic Support: {support}, Resistance: {resistance}")

        # Entry conditions: RSI-based signals
        if rsi_1m < 30 and position == 0:  # Oversold, buy call
            position = 1
            entry_price = close_price
            entry_time = data_1m.index[-1]
            strike_price = choose_strike('SPY', close_price)
            options_position = {'type': 'call', 'strike': strike_price}
            print(f"BUY CALL at {entry_price} (Strike: {strike_price}) on {entry_time}")
            alpaca.submit_order(
                symbol='SPY',
                qty=1,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
        
        elif rsi_1m > 70 and position == 0:  # Overbought, buy put
            position = -1
            entry_price = close_price
            entry_time = data_1m.index[-1]
            strike_price = choose_strike('SPY', close_price)
            options_position = {'type': 'put', 'strike': strike_price}
            print(f"BUY PUT at {entry_price} (Strike: {strike_price}) on {entry_time}")
            alpaca.submit_order(
                symbol='SPY',
                qty=1,
                side='buy',
                type='market',
                time_in_force='gtc'
            )

        # Average down if price continues to move against the position
        if position != 0 and options_position is not None:
            if position == 1 and close_price < entry_price * 0.98:  # Price falls 2% for averaging down
                print(f"Averaging down on CALL. Price: {close_price}, Entry Price: {entry_price}")
                new_strike_price = choose_strike('SPY', close_price)
                alpaca.submit_order(
                    symbol='SPY',
                    qty=1,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                entry_price = (entry_price + close_price) / 2  # Adjust break-even price

            elif position == -1 and close_price > entry_price * 1.02:  # Price rises 2% for averaging down
                print(f"Averaging down on PUT. Price: {close_price}, Entry Price: {entry_price}")
                new_strike_price = choose_strike('SPY', close_price)
                alpaca.submit_order(
                    symbol='SPY',
                    qty=1,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                entry_price = (entry_price + close_price) / 2  # Adjust break-even price

        # Exit conditions based on RSI and key price levels
        if position != 0 and options_position is not None:
            if (rsi_5m > 30 and rsi_5m < 70) or (rsi_10m > 30 and rsi_10m < 70) or (rsi_15m > 30 and rsi_15m < 70):
                # Neutral RSI on higher timeframes, exit position
                exit_price = close_price
                profit_loss += (exit_price - entry_price) * position
                print(f"Exit position at {exit_price} on {data_1m.index[-1]} with P/L: {profit_loss}")
                alpaca.submit_order(
                    symbol='SPY',
                    qty=1,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                position = 0
                options_position = None

            elif position == 1 and rsi_1m > 70:  # Sell call when RSI is in danger zone (above 70)
                exit_price = close_price
                profit_loss += (exit_price - entry_price) * position
                print(f"Exit CALL at {exit_price} on {data_1m.index[-1]} with P/L: {profit_loss}")
                alpaca.submit_order(
                    symbol='SPY',
                    qty=1,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                position = 0
                options_position = None

            elif position == -1 and rsi_1m < 30:  # Sell put when RSI is in danger zone (below 30)
                exit_price = close_price
                profit_loss += (exit_price - entry_price) * position
                print(f"Exit PUT at {exit_price} on {data_1m.index[-1]} with P/L: {profit_loss}")
                alpaca.submit_order(
                    symbol='SPY',
                    qty=1,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                position = 0
                options_position = None

            # Key Price Level Break
            if close_price > resistance:
                print(f"Price broke resistance at {resistance} - consider selling calls")
                # Additional logic for breaking price levels can be added here

            elif close_price < support:
                print(f"Price broke support at {support} - consider selling puts")
                # Additional logic for breaking price levels can be added here

        # Sleep for 60 seconds to poll the market every minute
        time.sleep(60)

# Check if market is open before running strategy
if is_market_open():
    real_time_strategy()
else:
    print("Market is closed. Real-time data is not available.")
