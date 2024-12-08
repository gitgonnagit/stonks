import yfinance as yf
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import datetime
from scipy.stats import norm

# Functions for Options Pricing
def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes formula for options pricing.
    Args:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to expiration in years.
        r (float): Risk-free rate.
        sigma (float): Volatility (standard deviation of stock returns).
        option_type (str): "call" or "put".
    Returns:
        float: Option price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Functions for Data and Indicators
def download_historical_data(symbol, interval="1m", start=None, end=None):
    """
    Download historical price data using yfinance with start and end dates.
    """
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start, end=end, interval=interval)
    if data.empty:
        raise ValueError("No data fetched. Please check the symbol and date range.")
    data.reset_index(inplace=True)
    data.rename(columns={"Close": "close"}, inplace=True)
    return data

def calculate_rsi(data, period=14):
    """Calculate RSI using TA-Lib."""
    return talib.RSI(data['close'].values, timeperiod=period)

def calculate_support_resistance(data, window=30):
    """Calculate dynamic support and resistance."""
    high = data['close'].rolling(window=window).max().iloc[-1]
    low = data['close'].rolling(window=window).min().iloc[-1]
    return low, high

# Backtest Strategy
def backtest_strategy(data, initial_capital=10000, rsi_period=14, support_resistance_window=30, slippage=0.001, risk_free_rate=0.02, volatility=0.2):
    """
    Backtest the RSI-based strategy with options pricing simulation.
    """
    capital = initial_capital
    position = 0  # 0 for no position, 1 for call, -1 for put
    entry_price = 0
    profit_loss = 0
    capital_history = []
    trade_log = []

    data['RSI'] = calculate_rsi(data, period=rsi_period)
    
    for i in range(len(data)):
        current_price = data['close'].iloc[i]
        current_rsi = data['RSI'].iloc[i]

        # Calculate support/resistance
        if i >= support_resistance_window:
            support, resistance = calculate_support_resistance(data.iloc[i-support_resistance_window:i])
        else:
            support, resistance = np.nan, np.nan
        
        # Options price simulation
        time_to_expiration = 1 / 252  # 1 day as a fraction of 1 year
        strike_price = round(current_price) + (1 if position == 0 else 0)
        option_price = black_scholes(current_price, strike_price, time_to_expiration, risk_free_rate, volatility, option_type="call" if position == 1 else "put")

        # Entry logic
        if position == 0:
            if current_rsi < 30:
                position = 1  # Buy call
                entry_price = option_price * (1 + slippage)
                print(f"BUY CALL at ${entry_price:.2f} (Strike: ${strike_price}) on {data['Datetime'].iloc[i]}")
            elif current_rsi > 70:
                position = -1  # Buy put
                entry_price = option_price * (1 + slippage)
                print(f"BUY PUT at ${entry_price:.2f} (Strike: ${strike_price}) on {data['Datetime'].iloc[i]}")

        # Exit logic
        elif position == 1:  # For call position
            if current_rsi > 70 or (not np.isnan(support) and current_price < support):
                exit_price = option_price * (1 - slippage)
                profit = exit_price - entry_price
                capital += profit
                profit_loss += profit
                print(f"SELL CALL at ${exit_price:.2f} | P/L: ${profit:.2f}")
                position = 0
                trade_log.append({"Type": "Call", "P/L": profit, "Capital": capital})
        elif position == -1:  # For put position
            if current_rsi < 30 or (not np.isnan(resistance) and current_price > resistance):
                exit_price = option_price * (1 - slippage)
                profit = entry_price - exit_price
                capital += profit
                profit_loss += profit
                print(f"SELL PUT at ${exit_price:.2f} | P/L: ${profit:.2f}")
                position = 0
                trade_log.append({"Type": "Put", "P/L": profit, "Capital": capital})

        # Record capital for plotting
        capital_history.append(capital)

    # Performance Metrics
    max_drawdown = max(1 - np.array(capital_history) / np.maximum.accumulate(capital_history))
    sharpe_ratio = (np.mean(np.diff(capital_history)) / np.std(np.diff(capital_history))) * np.sqrt(252)
    win_rate = sum(1 for t in trade_log if t["P/L"] > 0) / len(trade_log) if trade_log else 0

    print(f"\nFinal Capital: ${capital:.2f}")
    print(f"Total Profit/Loss: ${profit_loss:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Win Rate: {win_rate:.2%}")

    return capital, profit_loss, capital_history, trade_log

def main():
    # Define the backtest period
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=7)  # Last 7 days
    
    # Download historical data
    symbol = "SPY"
    try:
        data = download_historical_data(symbol, interval="1m", start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    except ValueError as e:
        print(e)
        return
    
    # Ensure the 'Datetime' column exists
    if 'Datetime' not in data.columns:
        data.rename(columns={"Datetime": "Datetime"}, inplace=True)
    
    # Run backtest
    final_capital, total_profit_loss, capital_history, trade_log = backtest_strategy(data)
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    plt.plot(capital_history, label="Capital Over Time")
    plt.title(f"Backtest Results for {symbol}")
    plt.xlabel("Time (1m intervals)")
    plt.ylabel("Capital ($)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
