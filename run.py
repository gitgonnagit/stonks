import backtrader as bt
import yfinance as yf

# Download stock data using yfinance
ticker = 'AAPL'  # Example: Apple stock
data = yf.download(ticker, start="2020-01-01", end="2024-12-31")

# Flatten the MultiIndex columns
data.columns = [col[0] for col in data.columns]  # Extract the first element of each tuple

# Custom Data Feed Class
class YFinanceData(bt.feeds.PandasData):
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None)
    )

# Define a simple moving average crossover strategy
class SimpleStrategy(bt.Strategy):
    # Initializing trade_count as an instance variable
    def __init__(self):
        self.trade_count = 0  # Counter for the number of trades per strategy instance
        # Define the moving averages
        self.short_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=5)  # Shorter period for faster crossovers
        self.long_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=20)  # Longer period for slower crossovers
        # Track the crossover
        self.cross = bt.indicators.CrossOver(self.short_ma, self.long_ma)

    def next(self):
        # Print current cash balance at each step
        print(f"Cash Balance: {self.broker.get_cash()}")
        
        # Logic for executing trades
        if self.cross > 0 and not self.position:
            self.buy()  # Enter position (buy)
            self.trade_count += 1  # Increment trade count
            print(f"BUY Signal - Trade {self.trade_count}")
        
        elif self.cross < 0 and self.position:
            self.sell()  # Exit position (sell)
            self.trade_count += 1  # Increment trade count
            print(f"SELL Signal - Trade {self.trade_count}")

# Create a Cerebro engine
cerebro = bt.Cerebro()

# Add strategy to the engine
cerebro.addstrategy(SimpleStrategy)

# Pass the DataFrame into Backtrader feed
data_feed = YFinanceData(dataname=data)

# Add data to the engine
cerebro.adddata(data_feed)

# Set initial cash and other parameters
cerebro.broker.set_cash(10000)
cerebro.broker.setcommission(commission=0.001)

# Run the backtest
print("Starting Backtest...\n")
cerebro.run()

# Print the final cash balance and total trades after the backtest
final_cash = cerebro.broker.get_cash()
final_trade_count = sum([strat[0].trade_count for strat in cerebro.strats])  # Accessing the first strategy in the tuple
print(f"\nFinal Cash Balance: {final_cash}")
print(f"Total Trades Executed: {final_trade_count}")
