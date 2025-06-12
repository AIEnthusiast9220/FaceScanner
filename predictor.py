import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import streamlit as st

class StockPredictor:
    """
    A class to predict stock prices using simple machine learning models
    """
    
    def __init__(self):
        """
        Initialize the StockPredictor
        """
        self.models = {}
        self.scalers = {}
        
    def prepare_features(self, stock_data, lookback_days=10):
        """
        Prepare features for machine learning models
        
        Args:
            stock_data (pd.DataFrame): Historical stock data
            lookback_days (int): Number of previous days to use as features
            
        Returns:
            tuple: (X, y) features and target arrays
        """
        try:
            if stock_data is None or len(stock_data) < lookback_days + 5:
                return None, None
            
            # Create feature matrix
            features = []
            targets = []
            
            # Technical indicators as features
            stock_data = stock_data.copy()
            
            # Price-based features
            stock_data['returns'] = stock_data['Close'].pct_change()
            stock_data['high_low_ratio'] = stock_data['High'] / stock_data['Low']
            stock_data['close_open_ratio'] = stock_data['Close'] / stock_data['Open']
            
            # Volume features
            stock_data['volume_ma'] = stock_data['Volume'].rolling(window=5).mean()
            stock_data['volume_ratio'] = stock_data['Volume'] / stock_data['volume_ma']
            
            # Moving averages
            stock_data['ma_5'] = stock_data['Close'].rolling(window=5).mean()
            stock_data['ma_10'] = stock_data['Close'].rolling(window=10).mean()
            stock_data['price_ma5_ratio'] = stock_data['Close'] / stock_data['ma_5']
            stock_data['price_ma10_ratio'] = stock_data['Close'] / stock_data['ma_10']
            
            # Volatility
            stock_data['volatility'] = stock_data['returns'].rolling(window=5).std()
            
            # Create lagged features
            feature_columns = [
                'Close', 'Volume', 'returns', 'high_low_ratio', 'close_open_ratio',
                'volume_ratio', 'price_ma5_ratio', 'price_ma10_ratio', 'volatility'
            ]
            
            # Remove rows with NaN values
            stock_data = stock_data.dropna()
            
            if len(stock_data) < lookback_days + 1:
                return None, None
            
            for i in range(lookback_days, len(stock_data)):
                # Features: previous lookback_days of data
                feature_row = []
                for j in range(lookback_days):
                    idx = i - lookback_days + j
                    for col in feature_columns:
                        if col in stock_data.columns:
                            feature_row.append(stock_data[col].iloc[idx])
                
                features.append(feature_row)
                targets.append(stock_data['Close'].iloc[i])
            
            return np.array(features), np.array(targets)
            
        except Exception as e:
            st.error(f"Error preparing features: {str(e)}")
            return None, None
    
    def train_model(self, X, y, model_type='linear'):
        """
        Train a prediction model
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target array
            model_type (str): Type of model to train
            
        Returns:
            tuple: (model, scaler, metrics)
        """
        try:
            if X is None or y is None or len(X) == 0:
                return None, None, {}
            
            # Split data into train and test
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if model_type == 'linear':
                model = LinearRegression()
            else:
                model = LinearRegression()  # Default fallback
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            metrics = {
                'train_r2': r2_score(y_train, train_pred),
                'test_r2': r2_score(y_test, test_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'test_size': len(y_test)
            }
            
            return model, scaler, metrics
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None, None, {}
    
    def predict_future_price(self, model, scaler, stock_data, days_ahead=7, lookback_days=10):
        """
        Predict future stock price
        
        Args:
            model: Trained model
            scaler: Fitted scaler
            stock_data (pd.DataFrame): Historical stock data
            days_ahead (int): Number of days to predict ahead
            lookback_days (int): Number of previous days to use as features
            
        Returns:
            dict: Prediction results
        """
        try:
            if model is None or scaler is None or stock_data is None:
                return None
            
            # Prepare the most recent data for prediction
            stock_data_copy = stock_data.copy()
            
            # Calculate same features as in training
            stock_data_copy['returns'] = stock_data_copy['Close'].pct_change()
            stock_data_copy['high_low_ratio'] = stock_data_copy['High'] / stock_data_copy['Low']
            stock_data_copy['close_open_ratio'] = stock_data_copy['Close'] / stock_data_copy['Open']
            
            stock_data_copy['volume_ma'] = stock_data_copy['Volume'].rolling(window=5).mean()
            stock_data_copy['volume_ratio'] = stock_data_copy['Volume'] / stock_data_copy['volume_ma']
            
            stock_data_copy['ma_5'] = stock_data_copy['Close'].rolling(window=5).mean()
            stock_data_copy['ma_10'] = stock_data_copy['Close'].rolling(window=10).mean()
            stock_data_copy['price_ma5_ratio'] = stock_data_copy['Close'] / stock_data_copy['ma_5']
            stock_data_copy['price_ma10_ratio'] = stock_data_copy['Close'] / stock_data_copy['ma_10']
            
            stock_data_copy['volatility'] = stock_data_copy['returns'].rolling(window=5).std()
            
            # Remove NaN values
            stock_data_copy = stock_data_copy.dropna()
            
            if len(stock_data_copy) < lookback_days:
                return None
            
            # Get the most recent features
            feature_columns = [
                'Close', 'Volume', 'returns', 'high_low_ratio', 'close_open_ratio',
                'volume_ratio', 'price_ma5_ratio', 'price_ma10_ratio', 'volatility'
            ]
            
            # Create feature vector for prediction
            feature_row = []
            for j in range(lookback_days):
                idx = len(stock_data_copy) - lookback_days + j
                for col in feature_columns:
                    if col in stock_data_copy.columns:
                        feature_row.append(stock_data_copy[col].iloc[idx])
            
            # Scale features and predict
            X_pred = scaler.transform([feature_row])
            predicted_price = model.predict(X_pred)[0]
            
            current_price = stock_data_copy['Close'].iloc[-1]
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100 if current_price > 0 else 0
            
            return {
                'price': predicted_price,
                'current_price': current_price,
                'change': price_change,
                'change_pct': price_change_pct,
                'days_ahead': days_ahead
            }
            
        except Exception as e:
            st.error(f"Error predicting future price: {str(e)}")
            return None
    
    def calculate_simple_ma_prediction(self, stock_data, days_ahead=7, ma_period=20):
        """
        Simple moving average based prediction as fallback
        
        Args:
            stock_data (pd.DataFrame): Historical stock data
            days_ahead (int): Number of days to predict ahead
            ma_period (int): Moving average period
            
        Returns:
            dict: Prediction results
        """
        try:
            if stock_data is None or len(stock_data) < ma_period:
                return None
            
            # Calculate moving average trend
            ma = stock_data['Close'].rolling(window=ma_period).mean()
            
            # Calculate trend (slope of MA)
            if len(ma) >= 5:
                recent_ma = ma.tail(5).values
                x = np.arange(len(recent_ma))
                trend = np.polyfit(x, recent_ma, 1)[0]  # Linear trend
            else:
                trend = 0
            
            current_price = stock_data['Close'].iloc[-1]
            current_ma = ma.iloc[-1]
            
            # Simple prediction: current price + trend * days
            predicted_price = current_price + (trend * days_ahead)
            
            # Add some bounds to make prediction more realistic
            max_change = current_price * 0.15  # Maximum 15% change
            predicted_price = max(current_price - max_change, 
                                min(current_price + max_change, predicted_price))
            
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100 if current_price > 0 else 0
            
            return {
                'price': predicted_price,
                'current_price': current_price,
                'change': price_change,
                'change_pct': price_change_pct,
                'days_ahead': days_ahead
            }
            
        except Exception as e:
            st.error(f"Error in simple MA prediction: {str(e)}")
            return None
    
    def predict_prices(self, stock_data):
        """
        Generate both short-term and long-term price predictions
        
        Args:
            stock_data (pd.DataFrame): Historical stock data
            
        Returns:
            dict: Dictionary containing short-term and long-term predictions
        """
        try:
            if stock_data is None or len(stock_data) < 30:
                st.error("Insufficient data for predictions. Need at least 30 days of data.")
                return None
            
            predictions = {}
            
            # Prepare features for ML model
            X, y = self.prepare_features(stock_data)
            
            if X is not None and y is not None and len(X) > 20:
                # Train ML model
                model, scaler, metrics = self.train_model(X, y)
                
                if model is not None and metrics.get('test_r2', 0) > 0.1:  # Reasonable R²
                    # Short-term prediction (7 days) using ML
                    short_pred = self.predict_future_price(model, scaler, stock_data, days_ahead=7)
                    if short_pred:
                        short_pred['confidence'] = min(95, max(50, metrics.get('test_r2', 0.5) * 100))
                        short_pred['method'] = 'ML Model'
                        predictions['short_term'] = short_pred
                    
                    # Long-term prediction (30 days) using ML
                    long_pred = self.predict_future_price(model, scaler, stock_data, days_ahead=30)
                    if long_pred:
                        # Lower confidence for longer predictions
                        long_pred['confidence'] = min(85, max(40, metrics.get('test_r2', 0.4) * 80))
                        long_pred['method'] = 'ML Model'
                        predictions['long_term'] = long_pred
            
            # Fallback to simple moving average predictions if ML fails
            if 'short_term' not in predictions:
                short_pred = self.calculate_simple_ma_prediction(stock_data, days_ahead=7)
                if short_pred:
                    short_pred['confidence'] = 65.0
                    short_pred['method'] = 'Moving Average'
                    predictions['short_term'] = short_pred
            
            if 'long_term' not in predictions:
                long_pred = self.calculate_simple_ma_prediction(stock_data, days_ahead=30)
                if long_pred:
                    long_pred['confidence'] = 55.0
                    long_pred['method'] = 'Moving Average'
                    predictions['long_term'] = long_pred
            
            return predictions
            
        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
            return None
    
    def calculate_targets(self, stock_data, predictions):
        """
        Calculate buy/sell targets based on predictions and technical analysis
        
        Args:
            stock_data (pd.DataFrame): Historical stock data
            predictions (dict): Price predictions
            
        Returns:
            dict: Buy/sell targets for short-term and long-term
        """
        try:
            if not predictions or stock_data is None:
                return {}
            
            current_price = stock_data['Close'].iloc[-1]
            targets = {}
            
            # Calculate volatility for target adjustment
            returns = stock_data['Close'].pct_change().dropna()
            volatility = returns.std() if len(returns) > 0 else 0.02
            
            # Short-term targets
            if 'short_term' in predictions:
                short_pred = predictions['short_term']
                predicted_price = short_pred['price']
                
                # Support and resistance levels (simplified)
                recent_high = stock_data['High'].tail(20).max()
                recent_low = stock_data['Low'].tail(20).min()
                
                # Calculate targets based on prediction and technical levels
                if predicted_price > current_price:
                    # Bullish prediction
                    sell_target = min(recent_high * 1.02, predicted_price * 1.05)
                    buy_target = max(current_price * 0.98, current_price - (volatility * current_price * 2))
                else:
                    # Bearish prediction
                    sell_target = max(current_price * 1.02, current_price + (volatility * current_price * 1))
                    buy_target = max(recent_low * 0.98, predicted_price * 0.95)
                
                targets['short_term'] = {
                    'buy': buy_target,
                    'sell': sell_target,
                    'current': current_price,
                    'predicted': predicted_price
                }
            
            # Long-term targets
            if 'long_term' in predictions:
                long_pred = predictions['long_term']
                predicted_price = long_pred['price']
                
                # Wider support and resistance for long-term
                long_high = stock_data['High'].tail(60).max()
                long_low = stock_data['Low'].tail(60).min()
                
                if predicted_price > current_price:
                    # Bullish long-term
                    sell_target = min(long_high * 1.08, predicted_price * 1.10)
                    buy_target = max(current_price * 0.95, current_price - (volatility * current_price * 3))
                else:
                    # Bearish long-term
                    sell_target = max(current_price * 1.05, current_price + (volatility * current_price * 2))
                    buy_target = max(long_low * 0.95, predicted_price * 0.90)
                
                targets['long_term'] = {
                    'buy': buy_target,
                    'sell': sell_target,
                    'current': current_price,
                    'predicted': predicted_price
                }
            
            return targets
            
        except Exception as e:
            st.error(f"Error calculating targets: {str(e)}")
            return {}
    
    def calculate_confidence_intervals(self, predictions, stock_data):
        """
        Calculate confidence intervals for predictions
        
        Args:
            predictions (dict): Price predictions
            stock_data (pd.DataFrame): Historical stock data
            
        Returns:
            dict: Confidence intervals for predictions
        """
        try:
            if not predictions or stock_data is None:
                return {}
            
            # Calculate historical volatility
            returns = stock_data['Close'].pct_change().dropna()
            volatility = returns.std() if len(returns) > 0 else 0.02
            
            intervals = {}
            
            for term, pred in predictions.items():
                if pred and 'price' in pred:
                    predicted_price = pred['price']
                    days_ahead = pred.get('days_ahead', 7)
                    
                    # Volatility adjustment based on time horizon
                    time_adjusted_vol = volatility * np.sqrt(days_ahead)
                    
                    # Calculate confidence intervals (assuming normal distribution)
                    confidence_95 = 1.96 * time_adjusted_vol * predicted_price
                    confidence_68 = 1.0 * time_adjusted_vol * predicted_price
                    
                    intervals[term] = {
                        'upper_95': predicted_price + confidence_95,
                        'lower_95': predicted_price - confidence_95,
                        'upper_68': predicted_price + confidence_68,
                        'lower_68': predicted_price - confidence_68,
                        'predicted': predicted_price
                    }
            
            return intervals
            
        except Exception as e:
            st.error(f"Error calculating confidence intervals: {str(e)}")
            return {}
