import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class LSTMModel:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    def _prepare_data(self, data, target_col='close'):
        """
        Prepare data for LSTM model
        
        Parameters:
        - data: DataFrame with features
        - target_col: Target column to predict
        
        Returns:
        - X: Input sequences
        - y: Target values
        """
        try:
            # Ensure data is a DataFrame
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Input data must be a pandas DataFrame")
            
            # Ensure target column exists
            if target_col not in data.columns:
                raise ValueError(f"Target column '{target_col}' not found in data")
            
            # Drop NaN values
            data = data.dropna()
            
            if len(data) < self.sequence_length:
                raise ValueError(f"Not enough data points. Need at least {self.sequence_length} points")
            
            # Scale the data
            scaled_data = self.scaler.fit_transform(data[[target_col]].values.reshape(-1, 1))
            
            X, y = [], []
            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i-self.sequence_length:i, 0])
                y.append(scaled_data[i, 0])
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Validate shapes
            if len(X.shape) != 2:
                raise ValueError(f"Unexpected X shape before reshape: {X.shape}")
            
            # Reshape X to (samples, time steps, features)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            return X, y
            
        except Exception as e:
            raise ValueError(f"Error preparing data: {str(e)}")
    
    def build_model(self, input_shape):
        """
        Build LSTM model
        
        Parameters:
        - input_shape: Shape of input data (sequence_length, features)
        
        Returns:
        - Compiled Keras model
        """
        model = Sequential()
        
        # LSTM layers
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
          # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        self.model = model
        return model
        
    def train(self, data, target_col='close', test_size=0.1, epochs=50, batch_size=32): # Changed test_size from 0.2 to 0.1
        """
        Train LSTM model
        
        Parameters:
        - data: DataFrame with features
        - target_col: Target column to predict
        - test_size: Proportion of data to use for testing (e.g., 0.1 for 10% test data)
        - epochs: Number of training epochs
        - batch_size: Batch size for training
        
        Returns:
        - Training history
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Validate inputs
            if test_size <= 0 or test_size >= 1:
                raise ValueError("test_size must be between 0 and 1")
            if epochs <= 0:
                raise ValueError("epochs must be positive")
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")
            
            # Prepare data
            logger.info(f"Preparing data for training, target column: {target_col}")
            X, y = self._prepare_data(data, target_col)
            logger.info(f"Data shape after preparation: X={X.shape}, y={y.shape}")
            
            # Split into train and test sets
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
            
            # Build model if not already built
            if self.model is None:
                logger.info(f"Building model with input shape: ({X_train.shape[1]}, {X_train.shape[2]})")
                self.build_model((X_train.shape[1], X_train.shape[2]))
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train the model
            logger.info("Starting model training...")
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=[early_stopping],
                verbose=0
            )
            
            logger.info("Model training completed successfully")
            return history
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    def predict(self, data, target_col='close', days_to_predict=30):
        """
        Make predictions
        
        Parameters:
        - data: DataFrame with features
        - target_col: Target column to predict
        - days_to_predict: Number of days to predict into the future
        
        Returns:
        - Array of predicted values
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare the last sequence from the data
        data = data.dropna()
        scaled_data = self.scaler.transform(data[[target_col]])
        
        # Use the last sequence_length data points for prediction
        last_sequence = scaled_data[-self.sequence_length:]
        last_sequence = np.reshape(last_sequence, (1, self.sequence_length, 1))
        
        # Make predictions for the specified number of days
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days_to_predict):
            # Predict the next value
            next_pred = self.model.predict(current_sequence)[0, 0]
            predictions.append(next_pred)
            
            # Update the sequence for the next prediction
            # Reshape next_pred to match the dimensions of current_sequence
            next_pred_reshaped = np.array([[[next_pred]]])
            current_sequence = np.append(current_sequence[:, 1:, :], next_pred_reshaped, axis=1)
        
        # Inverse transform to get actual values
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def train_and_evaluate(self, data, prediction_days, target_col='close', epochs=50, batch_size=32):
        """
        Train on historical data excluding the last n days, then predict those days and compare
        
        Parameters:
        - data: DataFrame with features
        - prediction_days: Number of days to exclude from training and use for evaluation
        - target_col: Target column to predict
        - epochs: Number of training epochs
        - batch_size: Batch size for training
        
        Returns:
        - Dictionary with predictions, actual values, and evaluation metrics
        """
        # Drop NaN values
        data = data.dropna()
        
        # Split data: training data and evaluation data (last n days)
        train_data = data.iloc[:-prediction_days].copy()
        eval_data = data.iloc[-prediction_days-self.sequence_length:].copy()
        
        # Scale the data using the training data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(train_data[[target_col]])
        
        # Train the model on historical data (excluding last n days)
        self.train(train_data, target_col=target_col, epochs=epochs, batch_size=batch_size)
        
        # Prepare the last sequence from the training data
        scaled_eval_data = self.scaler.transform(eval_data[[target_col]])
        input_sequence = scaled_eval_data[:self.sequence_length]
        input_sequence = np.reshape(input_sequence, (1, self.sequence_length, 1))
        
        # Make predictions for the evaluation period
        predictions = []
        current_sequence = input_sequence.copy()
        
        for _ in range(prediction_days):
            # Predict the next value
            next_pred = self.model.predict(current_sequence, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Update the sequence for the next prediction
            next_pred_reshaped = np.array([[[next_pred]]])
            current_sequence = np.append(current_sequence[:, 1:, :], next_pred_reshaped, axis=1)
        
        # Get actual values for comparison
        actual_values = eval_data.iloc[-prediction_days:][target_col].values
        
        # Inverse transform predictions to get actual values
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        # Calculate evaluation metrics
        mse = np.mean((predictions - actual_values) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
        
        return {
            'predictions': predictions,
            'actual_values': actual_values,
            'dates': eval_data.iloc[-prediction_days:].index,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mape': mape
            }
        }