from dash import Dash, html, dcc, Input, Output, State, exceptions, callback_context
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import os
import sys
import logging
import traceback
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc  # Add this import
import json  # Add this import for JSON parsing

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from data.collector import DataCollector
from features.technical_indicators import TechnicalFeatureGenerator
from models.lstm_model import LSTMModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the LSTM model
model = LSTMModel()

# Initialize the Dash app with Bootstrap
app = Dash(__name__,
           title="Stock Market Analyzer",
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True)

# Define popular stocks for quick selection
popular_stocks = [
    {"label": "Apple (AAPL)", "value": "AAPL"},
    {"label": "Microsoft (MSFT)", "value": "MSFT"},
    {"label": "Amazon (AMZN)", "value": "AMZN"},
    {"label": "Google (GOOGL)", "value": "GOOGL"},
    {"label": "Tesla (TSLA)", "value": "TSLA"},
    {"label": "NVIDIA (NVDA)", "value": "NVDA"},
    {"label": "Meta (META)", "value": "META"},
    {"label": "Netflix (NFLX)", "value": "NFLX"},
    {"label": "Disney (DIS)", "value": "DIS"},
    {"label": "JPMorgan Chase (JPM)", "value": "JPM"},
]

# Define the navigation bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Dashboard", href="#", id="nav-dashboard")),
        dbc.NavItem(dbc.NavLink("Technical Analysis", href="#", id="nav-technical")),
        dbc.NavItem(dbc.NavLink("Predictions", href="#", id="nav-predictions")),
        dbc.NavItem(dbc.NavLink("About", href="#", id="nav-about")),
    ],
    brand="Interactive Stock Market Analyzer",
    brand_href="#",
    color="primary",
    dark=True,
)

# Define the main dashboard page
dashboard_page = html.Div([
    # Error message display area
    dbc.Alert(id='error-display', color="danger", is_open=False, dismissable=True),
    
    # Stock selection and data fetching controls
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Select Stock Symbol:"),
                    dbc.InputGroup([
                        dbc.Input(id="stock-input", type="text", value="AAPL", placeholder="Enter stock symbol"),
                        dbc.Button("Fetch Data", id="fetch-button", color="success"),
                    ]),
                    html.Div([
                        html.Label("Popular Stocks:", className="mt-3"),
                        dbc.ListGroup(
                            [dbc.ListGroupItem(stock["label"], id={"type": "stock-item", "index": i}, 
                                             action=True, className="stock-list-item", 
                                             style={"cursor": "pointer"}) 
                             for i, stock in enumerate(popular_stocks)],
                            className="scrollable-stock-list"
                        ),
                    ]),
                ], width=4),
                dbc.Col([
                    html.Label("Time Period:"),
                    dbc.Select(
                        id="time-period-dropdown",
                        options=[
                            {"label": "1 Month", "value": "1mo"},
                            {"label": "3 Months", "value": "3mo"},
                            {"label": "6 Months", "value": "6mo"},
                            {"label": "1 Year", "value": "1y"},
                            {"label": "2 Years", "value": "2y"},
                            {"label": "5 Years", "value": "5y"},
                        ],
                        value="2y",
                    ),
                    html.Label("Prediction Days:", className="mt-3"),
                    dcc.Slider(
                        id="prediction-days-slider",
                        min=7,
                        max=60,
                        step=7,
                        value=30,
                        marks={i: f"{i}d" for i in range(7, 61, 7)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], width=8),
            ]),
        ])
    ], className="mb-4"),
    
    # Status message
    html.Div(id="status-message", className="text-danger mb-3"),
    
    # Main dashboard content
    dbc.Tabs([
        dbc.Tab(label="Price Chart", children=[
            dbc.Card([
                dbc.CardHeader(html.H4("Stock Price with Predictions", className="text-center")),
                dbc.CardBody([
                    dcc.Graph(id="price-prediction-chart", style={'height': '500px'}),
                ])
            ], className="mb-4"),
        ]),
        dbc.Tab(label="Technical Indicators", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Technical Indicators", className="text-center")),
                        dbc.CardBody([
                            dbc.Select(
                                id="indicator-dropdown",
                                options=[
                                    {"label": "RSI (14)", "value": "rsi_14"},
                                    {"label": "MACD", "value": "macd"},
                                    {"label": "Bollinger Bands Width", "value": "bb_width"},
                                    {"label": "Stochastic Oscillator", "value": "stoch_k"},
                                    {"label": "Volatility (20d)", "value": "volatility_20d"},
                                ],
                                value="rsi_14",
                                className="mb-3"
                            ),
                            dcc.Graph(id="technical-indicator-chart", style={'height': '300px'}),
                        ])
                    ]),
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Trading Volume", className="text-center")),
                        dbc.CardBody([
                            dcc.Graph(id="volume-chart", style={'height': '300px'}),
                        ])
                    ]),
                ], width=6),
            ], className="mb-4"),
        ]),
        dbc.Tab(label="Chart Analysis", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Candlestick Chart", className="text-center")),
                        dbc.CardBody([
                            dcc.Graph(id="candlestick-chart", style={'height': '400px'}),
                        ])
                    ]),
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Correlation Heatmap", className="text-center")),
                        dbc.CardBody([
                            dcc.Graph(id="correlation-heatmap", style={'height': '400px'}),
                        ])
                    ]),
                ], width=6),
            ], className="mb-4"),
        ]),
        dbc.Tab(label="Predictions", children=[
            dbc.Card([
                dbc.CardHeader(html.H4("Price Predictions", className="text-center")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="prediction-metrics", className="mb-4"),
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="prediction-table", className="table-responsive"),
                        ], width=12),
                    ]),
                ])
            ]),
        ]),
    ]),
    
    # Store the data in a hidden div for sharing between callbacks
    dcc.Store(id="stock-data-store"),
    dcc.Store(id="predictions-store"),
    dcc.Store(id="model-results-store") # Add new store for model results
])

# Define the app layout
# Define the app layout
app.layout = html.Div([
    navbar,
    dbc.Container([
        dashboard_page,
    ], fluid=True, className="mt-4")
])

# Callback for stock list item clicks
@app.callback(
    Output("stock-input", "value"),
    [Input({"type": "stock-item", "index": i}, "n_clicks") for i in range(len(popular_stocks))],
    prevent_initial_call=True
)
def update_stock_input(*args):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    
    # Get the id of the clicked item
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    index = json.loads(button_id)["index"]
    
    # Return the value of the selected stock
    return popular_stocks[index]["value"]

# Callback to update prediction metrics and store full model results
@app.callback(
    [Output("prediction-metrics", "children"),
     Output("model-results-store", "data")],  # Output to the new store
    [Input("stock-data-store", "data"),
     Input("prediction-days-slider", "value")],
    prevent_initial_call=True
)
def update_prediction_metrics_and_store_results(stock_data, prediction_days):
    if stock_data is None:
        return html.Div("No data available"), None # Return None for the store as well
    
    try:
        # Create and train the model
        df = pd.DataFrame({
            'dates': stock_data['dates'],
            'open': stock_data['open'],
            'high': stock_data['high'],
            'low': stock_data['low'],
            'close': stock_data['close'],
            'volume': stock_data['volume']
        })
        
        # Convert dates to datetime and set as index
        df['dates'] = pd.to_datetime(df['dates'])
        df = df.set_index('dates')
        
        # Initialize and train the model
        if len(df) < 60:  # Check df length instead of stock_features
            logger.warning(f"Not enough data points for LSTM model. Need at least 60, got {len(df)}")
            return html.Div("Not enough data points. Please select a longer time period.", className="alert alert-warning"), None
            
        results = model.train_and_evaluate(df, prediction_days=prediction_days)
        
        # Create status message with metrics
        metrics = results['metrics']
        metrics_card = dbc.Card([
            dbc.CardHeader(html.H5("Model Evaluation Metrics")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H3(f"{metrics['mse']:.4f}"),
                                html.P("Mean Squared Error (MSE)"),
                            ], className="text-center")
                        ])
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H3(f"{metrics['rmse']:.4f}"),
                                html.P("Root Mean Squared Error (RMSE)"),
                            ], className="text-center")
                        ])
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H3(f"{metrics['mape']:.2f}%"),
                                html.P("Mean Absolute Percentage Error (MAPE)"),
                            ], className="text-center")
                        ])
                    ], width=4),
                ])
            ])
        ])
        
        return metrics_card, results # Return full results to the store
        
    except Exception as e:
        logger.error(f"Error in prediction metrics callback: {str(e)}")
        logger.debug(traceback.format_exc())
        return html.Div("Error calculating prediction metrics. Check logs for details.", className="alert alert-danger"), None

# Update the prediction table callback
# Remove this duplicate callback function


# Keep the existing callbacks for other components
@app.callback(
    Output("prediction-table", "children"),
    [Input("model-results-store", "data")], # Input from the new store
    prevent_initial_call=True
)
def update_prediction_table(model_results): # Changed input from stock_data, prediction_days
    if model_results is None:
        return html.Div("No data available for prediction table.")
    
    try:
        # results are already computed and stored
        results = model_results
        
        # Create a comparison table
        table_header = [
            html.Thead(html.Tr([
                html.Th("Date"),
                html.Th("Actual Price"),
                html.Th("Predicted Price"),
                html.Th("Difference"),
                html.Th("% Error")
            ]))
        ]
        
        rows = []
        # Ensure all required keys are in results
        if not all(k in results for k in ['dates', 'actual_values', 'predictions']):
            logger.error(f"Missing keys in model_results for prediction table: {results.keys()}")
            return html.Div("Error: Incomplete model results for table.", className="alert alert-danger")

        for date, actual, pred in zip(results['dates'], results['actual_values'], results['predictions']):
            diff = pred - actual
            pct_error = (abs(diff) / actual) * 100 if actual != 0 else 0
            
            # Determine color based on error percentage
            color = "success" if pct_error < 5 else "warning" if pct_error < 10 else "danger"
            
            row = html.Tr([
                html.Td(date),
                html.Td(f"${actual:.2f}"),
                html.Td(f"${pred:.2f}"),
                html.Td(f"${diff:.2f}", style={"color": "green" if diff >= 0 else "red"}),
                html.Td(f"{pct_error:.2f}%", className=f"text-{color}")
            ])
            rows.append(row)
        
        table_body = [html.Tbody(rows)]
        
        table = dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True, striped=True)
        
        return table
        
    except Exception as e:
        logger.error(f"Error in prediction table callback: {str(e)}")
        logger.debug(traceback.format_exc())
        return html.Div("Error generating prediction table. Check logs for details.")

# Add this callback for the fetch button
@app.callback(
    [Output("stock-data-store", "data"),
     Output("status-message", "children"),
     Output("error-display", "is_open"),
     Output("error-display", "children")],
    [Input("fetch-button", "n_clicks")],
    [State("stock-input", "value"),
     State("time-period-dropdown", "value")],
    prevent_initial_call=True
)
def fetch_stock_data(n_clicks, stock_symbol, time_period):
    if n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    try:
        # Initialize the data collector
        collector = DataCollector()
        
        # Convert time period to days
        period_map = {
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825
        }
        days = period_map.get(time_period, 730)  # Default to 2 years
        
        # Get the stock data
        stock_data = collector.get_stock_data(stock_symbol, days=days)
        
        if stock_data is None or stock_data.empty:
            return None, f"No data found for {stock_symbol}", True, f"Failed to fetch data for {stock_symbol}. Please check the symbol and try again."
        
        # Process the data for storage
        data_dict = {
            "dates": stock_data.index.strftime("%Y-%m-%d").tolist(),
            "open": stock_data["Open"].tolist(),
            "high": stock_data["High"].tolist(),
            "low": stock_data["Low"].tolist(),
            "close": stock_data["Close"].tolist(),
            "volume": stock_data["Volume"].tolist()
        }
        
        return data_dict, f"Data for {stock_symbol} loaded successfully!", False, ""
        
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        logger.debug(traceback.format_exc())
        return None, "", True, f"Error: {str(e)}"

# Callback for Price Prediction Chart
@app.callback(
    Output("price-prediction-chart", "figure"),
    [Input("stock-data-store", "data"),
     Input("model-results-store", "data")],
    prevent_initial_call=True
)
def update_price_prediction_chart(stock_data_json, model_results_json):
    if not stock_data_json or not model_results_json:
        logger.info("Price prediction chart: Not enough data to plot.")
        return go.Figure()

    try:
        # Historical data
        hist_dates = pd.to_datetime(stock_data_json['dates'])
        hist_close = stock_data_json['close']

        # Model results for predictions
        pred_dates = pd.to_datetime(model_results_json['dates'])
        actual_values = model_results_json['actual_values'] # Actual values during prediction period
        predictions = model_results_json['predictions']

        fig = go.Figure()

        # Plot historical actual prices
        fig.add_trace(go.Scatter(x=hist_dates, y=hist_close, mode='lines', name='Historical Close Price'))

        # Plot actual prices during the prediction period (if they align with prediction dates)
        # This assumes model_results_json['dates'] are the dates for which predictions were made.
        # And historical data might overlap or precede this.
        # For simplicity, we plot actuals from model_results if available, otherwise historical close for those dates.
        
        # Find the portion of historical data that corresponds to the prediction period for comparison
        comparison_actual_values = []
        comparison_dates = []

        # Create a dictionary for quick lookup of historical close prices
        hist_data_map = {date: close for date, close in zip(hist_dates, hist_close)}

        for date in pred_dates:
            if date in hist_data_map:
                comparison_actual_values.append(hist_data_map[date])
                comparison_dates.append(date)
        
        if comparison_dates:
            fig.add_trace(go.Scatter(x=comparison_dates, y=comparison_actual_values, mode='lines', name='Actual Price (Prediction Period)', line=dict(dash='dot')))

        # Plot predicted prices
        fig.add_trace(go.Scatter(x=pred_dates, y=predictions, mode='lines', name='Predicted Price', line=dict(color='orange')))

        fig.update_layout(
            title="Stock Price with Predictions",
            xaxis_title="Date",
            yaxis_title="Price",
            legend_title="Legend"
        )
        logger.info("Price prediction chart updated.")
        return fig

    except Exception as e:
        logger.error(f"Error updating price prediction chart: {str(e)}")
        logger.debug(traceback.format_exc())
        return go.Figure()

@app.callback(
    Output("technical-indicator-chart", "figure"),
    [Input("stock-data-store", "data"), 
     Input("indicator-dropdown", "value")],
    prevent_initial_call=True
)
def update_technical_indicator_chart(stock_data_json, selected_indicator):
    if not stock_data_json:
        logger.info("Technical indicator chart: No stock data.")
        return go.Figure()
    
    try:
        df = pd.DataFrame({
            'Open': stock_data_json['open'],
            'High': stock_data_json['high'],
            'Low': stock_data_json['low'],
            'Close': stock_data_json['close'],
            'Volume': stock_data_json['volume']
        })
        df.index = pd.to_datetime(stock_data_json['dates'])

        # Rename columns to lowercase to match TechnicalFeatureGenerator expectations
        df.columns = [col.lower() for col in df.columns]

        feature_generator = TechnicalFeatureGenerator()
        
        # Calculate indicators - this might add multiple columns
        df_with_indicators = feature_generator.add_all_features(df.copy()) # Use a copy to avoid modifying original df if not intended

        fig = go.Figure()

        if selected_indicator == 'rsi_14' and 'rsi_14' in df_with_indicators.columns:
            fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['rsi_14'], mode='lines', name='RSI (14)'))
            fig.update_layout(title="Relative Strength Index (RSI)", yaxis_title="RSI")
        elif selected_indicator == 'macd':
            if 'macd' in df_with_indicators.columns:
                fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['macd'], mode='lines', name='MACD'))
            if 'macd_signal' in df_with_indicators.columns:
                fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['macd_signal'], mode='lines', name='MACD Signal'))
            if 'macd_hist' in df_with_indicators.columns:
                fig.add_trace(go.Bar(x=df_with_indicators.index, y=df_with_indicators['macd_hist'], name='MACD Histogram'))
            fig.update_layout(title="MACD", yaxis_title="Value")
        elif selected_indicator == 'bb_width' and 'bb_width' in df_with_indicators.columns:
            fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['bb_width'], mode='lines', name='Bollinger Bands Width'))
            fig.update_layout(title="Bollinger Bands Width", yaxis_title="Width")
        elif selected_indicator == 'stoch_k' and 'stoch_k' in df_with_indicators.columns:
            if 'stoch_k' in df_with_indicators.columns:
                 fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['stoch_k'], mode='lines', name='Stochastic %K'))
            if 'stoch_d' in df_with_indicators.columns: # Often plotted together
                 fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['stoch_d'], mode='lines', name='Stochastic %D'))
            fig.update_layout(title="Stochastic Oscillator", yaxis_title="Value")
        elif selected_indicator == 'volatility_20d' and 'volatility_20d' in df_with_indicators.columns:
            fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['volatility_20d'], mode='lines', name='Volatility (20d)'))
            fig.update_layout(title="Price Volatility (20-day)", yaxis_title="Volatility")
        else:
            logger.warning(f"Selected indicator '{selected_indicator}' not available or not implemented yet.")
            return go.Figure() # Return empty figure if indicator not found

        fig.update_layout(xaxis_title="Date")
        logger.info(f"Technical indicator chart updated for {selected_indicator}.")
        return fig

    except Exception as e:
        logger.error(f"Error updating technical indicator chart: {str(e)}")
        logger.debug(traceback.format_exc())
        return go.Figure()

@app.callback(
    Output("volume-chart", "figure"),
    [Input("stock-data-store", "data")],
    prevent_initial_call=True
)
def update_volume_chart(stock_data_json):
    if not stock_data_json:
        return go.Figure()
    try:
        dates = pd.to_datetime(stock_data_json['dates'])
        volume = stock_data_json['volume']
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dates, y=volume, name='Volume'))
        fig.update_layout(title="Trading Volume", xaxis_title="Date", yaxis_title="Volume")
        logger.info("Volume chart updated.")
        return fig
    except Exception as e:
        logger.error(f"Error updating volume chart: {str(e)}")
        return go.Figure()

@app.callback(
    Output("candlestick-chart", "figure"),
    [Input("stock-data-store", "data")],
    prevent_initial_call=True
)
def update_candlestick_chart(stock_data_json):
    if not stock_data_json:
        return go.Figure()
    try:
        dates = pd.to_datetime(stock_data_json['dates'])
        open_prices = stock_data_json['open']
        high_prices = stock_data_json['high']
        low_prices = stock_data_json['low']
        close_prices = stock_data_json['close']
        
        fig = go.Figure(data=[go.Candlestick(x=dates,
                                           open=open_prices,
                                           high=high_prices,
                                           low=low_prices,
                                           close=close_prices,
                                           name='Candlestick')])
        fig.update_layout(title="Candlestick Chart", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False)
        logger.info("Candlestick chart updated.")
        return fig
    except Exception as e:
        logger.error(f"Error updating candlestick chart: {str(e)}")
        return go.Figure()

@app.callback(
    Output("correlation-heatmap", "figure"),
    [Input("stock-data-store", "data")],
    prevent_initial_call=True
)
def update_correlation_heatmap(stock_data_json):
    if not stock_data_json:
        return go.Figure()
    try:
        df = pd.DataFrame({
            'Open': stock_data_json['open'],
            'High': stock_data_json['high'],
            'Low': stock_data_json['low'],
            'Close': stock_data_json['close'],
            'Volume': stock_data_json['volume']
        })
        correlation_matrix = df.corr()
        fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", title="Correlation Heatmap")
        logger.info("Correlation heatmap updated.")
        return fig
    except Exception as e:
        logger.error(f"Error updating correlation heatmap: {str(e)}")
        return go.Figure()

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=8050)