import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our modules
from data.collector import DataCollector
from features.technical_indicators import TechnicalFeatureGenerator
from models.lstm_model import LSTMModel

def main():
    # Initialize data collector
    collector = DataCollector()
    
    # Get historical data for a stock (e.g., Apple)
    print("Fetching historical data for AAPL...")
    aapl_data = collector.get_historical_data_yf("AAPL", period="5y")
    
    if aapl_data is None or aapl_data.empty:
        print("Failed to fetch data from Yahoo Finance. Trying Alpha Vantage...")
        # Try Alpha Vantage as a fallback if you have an API key
        if os.getenv("ALPHA_VANTAGE_API_KEY"):
            aapl_data = collector.get_historical_data_av("AAPL")
            if aapl_data is None or aapl_data.empty:
                print("Failed to fetch data from all sources. Using sample data for development.")
                aapl_data = collector.get_sample_data()
        else:
            print("No Alpha Vantage API key found. Using sample data for development.")
            aapl_data = collector.get_sample_data()
    
    print(f"Fetched {len(aapl_data)} data points for AAPL")
    
    # Generate technical features
    print("Generating technical features...")
    feature_generator = TechnicalFeatureGenerator()
    aapl_features = feature_generator.add_all_features(aapl_data)
    
    # Drop rows with NaN values (usually at the beginning due to rolling windows)
    aapl_features = aapl_features.dropna()
    
    print(f"Generated {len(aapl_features.columns) - 5} technical features")
    
    # Train LSTM model
    print("Training LSTM model...")
    model = LSTMModel(sequence_length=60)
    history = model.train(aapl_features, target_col='close', epochs=50)
    
    # Make predictions for the next 30 days
    print("Making predictions...")
    predictions = model.predict(aapl_features, target_col='close', days_to_predict=30)
    
    # Create dates for the predictions
    last_date = aapl_features.index[-1]
    prediction_dates = [last_date + timedelta(days=i+1) for i in range(len(predictions))]
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(aapl_features.index[-100:], aapl_features['close'][-100:], label='Historical Data')
    
    # Plot predictions
    plt.plot(prediction_dates, predictions, label='Predictions', color='red')
    
    plt.title('AAPL Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    # Save the plot
    plt.savefig('prediction_plot.png')
    plt.close()
    
    print("Prediction completed and saved to 'prediction_plot.png'")
    
    # Print the predicted prices
    print("\nPredicted prices for the next 30 days:")
    for date, price in zip(prediction_dates, predictions):
        print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")
    
    # After the existing code in main() function, before the return statement
    # Add these imports at the top of the file
    import seaborn as sns
    from matplotlib.dates import MonthLocator, DateFormatter
    import matplotlib.gridspec as gridspec
    
    # Create enhanced visualizations
    print("Creating enhanced visualizations...")
    
    # Set the style for seaborn
    sns.set(style="darkgrid")
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 2)
    
    # 1. Price and Prediction Plot (enhanced version of existing plot)
    ax1 = plt.subplot(gs[0, :])
    sns.lineplot(x=aapl_features.index[-100:], y=aapl_features['close'][-100:], label='Historical Data', ax=ax1)
    sns.lineplot(x=prediction_dates, y=predictions, label='Predictions', color='red', ax=ax1)
    ax1.set_title('AAPL Stock Price Prediction', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.xaxis.set_major_locator(MonthLocator())
    ax1.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. Volume Plot
    ax2 = plt.subplot(gs[1, 0])
    sns.barplot(x=aapl_features.index[-30:].strftime('%d-%m'), y=aapl_features['volume'][-30:], ax=ax2, alpha=0.7)
    ax2.set_title('Recent Trading Volume', fontsize=12)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume')
    ax2.tick_params(axis='x', rotation=90)
    
    # 3. Technical Indicators Plot
    ax3 = plt.subplot(gs[1, 1])
    sns.lineplot(x=aapl_features.index[-60:], y=aapl_features['rsi_14'][-60:], label='RSI', ax=ax3)
    ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax3.set_title('RSI Indicator (14-day)', fontsize=12)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('RSI Value')
    ax3.grid(True, alpha=0.3)
    
    # 4. Correlation Heatmap
    ax4 = plt.subplot(gs[2, 0])
    # Select a subset of features to avoid overcrowding
    selected_features = ['close', 'volume', 'rsi_14', 'macd', 'bb_width', 'volatility_20d']
    correlation = aapl_features[selected_features].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax4, fmt='.2f', cbar=False)
    ax4.set_title('Feature Correlation Heatmap', fontsize=12)
    
    # 5. Candlestick-like Plot for recent days
    ax5 = plt.subplot(gs[2, 1])
    recent_data = aapl_features[-15:]
    # Plot high-low range as vertical lines
    ax5.vlines(x=recent_data.index, ymin=recent_data['low'], ymax=recent_data['high'], color='black', linewidth=1)
    # Plot open-close range as colored bars
    for i, (idx, row) in enumerate(recent_data.iterrows()):
        if row['close'] >= row['open']:
            color = 'green'
        else:
            color = 'red'
        ax5.plot([idx, idx], [row['open'], row['close']], color=color, linewidth=4)
    ax5.set_title('Recent Price Action (Candlestick-like)', fontsize=12)
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Price ($)')
    ax5.tick_params(axis='x', rotation=45)
    
    # Adjust layout and save the enhanced visualization
    plt.tight_layout()
    plt.savefig('enhanced_visualizations.png', dpi=300)
    plt.close()
    
    print("Enhanced visualizations saved to 'enhanced_visualizations.png'")
    
    # Print the predicted prices
    print("\nPredicted prices for the next 30 days:")
    for date, price in zip(prediction_dates, predictions):
        print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")

if __name__ == "__main__":
    main()