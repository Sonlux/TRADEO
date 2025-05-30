import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice

class TechnicalFeatureGenerator:
    def __init__(self):
        pass
    
    def add_all_features(self, df):
        """
        Add all technical indicators to the dataframe
        
        Parameters:
        - df: DataFrame with OHLCV data
        
        Returns:
        - DataFrame with added technical indicators
        """
        if df is None or df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain all of these columns: {required_cols}")
        
        # Add moving averages
        df = self.add_moving_averages(df)
        
        # Add RSI
        df = self.add_rsi(df)
        
        # Add MACD
        df = self.add_macd(df)
        
        # Add Bollinger Bands
        df = self.add_bollinger_bands(df)
        
        # Add Stochastic Oscillator
        df = self.add_stochastic_oscillator(df)
        
        # Add volatility measures
        df = self.add_volatility_measures(df)
        
        # Add price momentum features
        df = self.add_momentum_features(df)
        
        return df
    
    def add_moving_averages(self, df, windows=[20, 50, 200]):
        """
        Add Simple and Exponential Moving Averages
        
        Parameters:
        - df: DataFrame with OHLCV data
        - windows: List of window sizes for moving averages
        
        Returns:
        - DataFrame with added moving averages
        """
        for window in windows:
            # Simple Moving Average
            sma = SMAIndicator(close=df['close'], window=window)
            df[f'sma_{window}'] = sma.sma_indicator()
            
            # Exponential Moving Average
            ema = EMAIndicator(close=df['close'], window=window)
            df[f'ema_{window}'] = ema.ema_indicator()
        
        return df
    
    def add_rsi(self, df, window=14):
        """
        Add Relative Strength Index
        
        Parameters:
        - df: DataFrame with OHLCV data
        - window: Window size for RSI calculation
        
        Returns:
        - DataFrame with added RSI
        """
        rsi = RSIIndicator(close=df['close'], window=window)
        df[f'rsi_{window}'] = rsi.rsi()
        return df
    
    def add_macd(self, df, window_slow=26, window_fast=12, window_sign=9):
        """
        Add Moving Average Convergence Divergence
        
        Parameters:
        - df: DataFrame with OHLCV data
        - window_slow: Window size for slow EMA
        - window_fast: Window size for fast EMA
        - window_sign: Window size for signal line
        
        Returns:
        - DataFrame with added MACD
        """
        macd = MACD(
            close=df['close'],
            window_slow=window_slow,
            window_fast=window_fast,
            window_sign=window_sign
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        return df
    
    def add_bollinger_bands(self, df, window=20, window_dev=2):
        """
        Add Bollinger Bands
        
        Parameters:
        - df: DataFrame with OHLCV data
        - window: Window size for moving average
        - window_dev: Number of standard deviations
        
        Returns:
        - DataFrame with added Bollinger Bands
        """
        bollinger = BollingerBands(
            close=df['close'],
            window=window,
            window_dev=window_dev
        )
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        return df
    
    def add_stochastic_oscillator(self, df, window=14, smooth_window=3):
        """
        Add Stochastic Oscillator
        
        Parameters:
        - df: DataFrame with OHLCV data
        - window: Window size for Stochastic Oscillator
        - smooth_window: Window size for moving average of %K
        
        Returns:
        - DataFrame with added Stochastic Oscillator
        """
        stoch = StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=window,
            smooth_window=smooth_window
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        return df
    
    def add_volatility_measures(self, df, windows=[5, 20, 60]):
        """
        Add volatility measures
        
        Parameters:
        - df: DataFrame with OHLCV data
        - windows: List of window sizes for volatility calculation
        
        Returns:
        - DataFrame with added volatility measures
        """
        # Calculate daily returns
        df['daily_return'] = df['close'].pct_change()
        
        for window in windows:
            # Standard deviation of returns (historical volatility)
            df[f'volatility_{window}d'] = df['daily_return'].rolling(window=window).std() * np.sqrt(252)  # Annualized
            
            # Average True Range (ATR)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f'atr_{window}'] = true_range.rolling(window=window).mean()
        
        return df
    
    def add_momentum_features(self, df, windows=[1, 5, 10, 20, 60]):
        """
        Add price momentum features
        
        Parameters:
        - df: DataFrame with OHLCV data
        - windows: List of window sizes for momentum calculation
        
        Returns:
        - DataFrame with added momentum features
        """
        for window in windows:
            # Price Rate of Change
            df[f'roc_{window}'] = df['close'].pct_change(periods=window) * 100
            
            # Price momentum (current price / price n periods ago)
            df[f'momentum_{window}'] = df['close'] / df['close'].shift(window)
        
        return df