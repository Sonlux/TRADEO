import os
import pandas as pd
import numpy as np  # Add this import
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta

class DataCollector:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if self.api_key:
            self.alpha_vantage = TimeSeries(key=self.api_key, output_format='pandas')
    
    def get_historical_data_yf(self, symbol, period="5y", interval="1d"):
        """
        Fetch historical stock data from Yahoo Finance
        
        Parameters:
        - symbol: Stock ticker symbol
        - period: Time period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        - interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
        - DataFrame with OHLCV data
        """
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)
            
            # Check if data is empty
            if data.empty:
                print(f"No data returned for {symbol}. Trying with a different period...")
                # Try with a different period as fallback
                data = stock.history(period="1y", interval=interval)
                
                if data.empty:
                    print(f"Still no data for {symbol}. Please check if the symbol is valid.")
                    return None
            
            # Rename columns to standard format
            data.columns = [col.lower() for col in data.columns]
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_historical_data_av(self, symbol, outputsize="full"):
        """
        Fetch historical stock data from Alpha Vantage
        
        Parameters:
        - symbol: Stock ticker symbol
        - outputsize: 'compact' (latest 100 data points) or 'full' (up to 20 years of data)
        
        Returns:
        - DataFrame with OHLCV data
        """
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")
        
        try:
            data, meta_data = self.alpha_vantage.get_daily(symbol=symbol, outputsize=outputsize)
            # Rename columns to standard format
            data.columns = [col.split(". ")[1].lower() for col in data.columns]
            return data
        except Exception as e:
            print(f"Error fetching data from Alpha Vantage for {symbol}: {e}")
            return None
    
    def get_fundamental_data(self, symbol):
        """
        Fetch fundamental company data from Yahoo Finance
        
        Parameters:
        - symbol: Stock ticker symbol
        
        Returns:
        - Dictionary with fundamental data
        """
        try:
            stock = yf.Ticker(symbol)
            # Get key statistics
            info = stock.info
            # Get financial data
            balance_sheet = stock.balance_sheet
            income_stmt = stock.income_stmt
            cash_flow = stock.cashflow
            
            return {
                "info": info,
                "balance_sheet": balance_sheet,
                "income_statement": income_stmt,
                "cash_flow": cash_flow
            }
        except Exception as e:
            print(f"Error fetching fundamental data for {symbol}: {e}")
            return None
    
    def get_multiple_stocks(self, symbols, period="5y", interval="1d"):
        """
        Fetch historical data for multiple stocks
        
        Parameters:
        - symbols: List of stock ticker symbols
        - period: Time period to fetch
        - interval: Data interval
        
        Returns:
        - Dictionary of DataFrames with OHLCV data for each symbol
        """
        result = {}
        for symbol in symbols:
            data = self.get_historical_data_yf(symbol, period, interval)
            if data is not None:
                result[symbol] = data
        return result
    
    def get_sample_data(self):
        """
        Return sample stock data for development when APIs fail
        """
        # Create a date range from 3 years ago to today
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)  # Roughly 3 years of data
        
        # Create a date range
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Create sample data
        np.random.seed(42)  # For reproducibility
        n = len(dates)
        close = 100 + np.cumsum(np.random.normal(0, 1, n))
        open_price = close - np.random.normal(0, 1, n)
        high = np.maximum(close, open_price) + np.random.normal(0, 0.5, n)
        low = np.minimum(close, open_price) - np.random.normal(0, 0.5, n)
        volume = np.random.randint(1000000, 10000000, n)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
        
        return df
    
    def get_stock_data(self, symbol, days=730):
        """
        Fetch historical stock data for a given symbol and number of days using Yahoo Finance.
        """
        period_map = {
            30: "1mo",
            90: "3mo",
            180: "6mo",
            365: "1y",
            730: "2y",
            1825: "5y"
        }
        # Find the closest period string for the given days
        period = period_map.get(days, "2y")
        data = self.get_historical_data_yf(symbol, period=period, interval="1d")
        if data is not None and not data.empty:
            # Standardize column names to match dashboard expectations
            data = data.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
        return data