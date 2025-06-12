import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class StockAnalyzer:
    """
    A class to analyze stock data using Yahoo Finance API
    """
    
    def __init__(self, symbol):
        """
        Initialize the StockAnalyzer with a stock symbol
        
        Args:
            symbol (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        """
        self.symbol = symbol.upper()
        self.ticker = None
        self.info = None
        
    def validate_symbol(self):
        """
        Validate if the stock symbol exists and has data
        
        Returns:
            bool: True if symbol is valid, False otherwise
        """
        try:
            self.ticker = yf.Ticker(self.symbol)
            # Try to get basic info to validate symbol
            info = self.ticker.info
            
            # Check if we got meaningful data
            if not info or len(info) < 5:
                return False
                
            # Try to get recent data to ensure symbol is tradeable
            hist = self.ticker.history(period="5d")
            if hist.empty:
                return False
                
            return True
            
        except Exception as e:
            st.error(f"Error validating symbol {self.symbol}: {str(e)}")
            return False
    
    def get_stock_data(self, period="1y"):
        """
        Fetch historical stock data
        
        Args:
            period (str): Time period for historical data
                         Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        
        Returns:
            pd.DataFrame: Historical stock data with OHLCV columns
        """
        try:
            if not self.ticker:
                self.ticker = yf.Ticker(self.symbol)
            
            # Fetch historical data
            hist = self.ticker.history(period=period)
            
            if hist.empty:
                st.error(f"No historical data available for {self.symbol}")
                return None
            
            # Clean the data
            hist = hist.dropna()
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in hist.columns for col in required_columns):
                st.error(f"Missing required data columns for {self.symbol}")
                return None
            
            return hist
            
        except Exception as e:
            st.error(f"Error fetching stock data for {self.symbol}: {str(e)}")
            return None
    
    def get_company_info(self):
        """
        Fetch company information and financial metrics
        
        Returns:
            dict: Company information and financial metrics
        """
        try:
            if not self.ticker:
                self.ticker = yf.Ticker(self.symbol)
            
            info = self.ticker.info
            
            if not info:
                st.warning(f"No company information available for {self.symbol}")
                return {}
            
            # Store info for later use
            self.info = info
            return info
            
        except Exception as e:
            st.error(f"Error fetching company info for {self.symbol}: {str(e)}")
            return {}
    
    def get_financial_ratios(self, stock_data):
        """
        Calculate additional financial ratios and technical indicators
        
        Args:
            stock_data (pd.DataFrame): Historical stock data
            
        Returns:
            dict: Dictionary containing calculated ratios and indicators
        """
        try:
            if stock_data is None or stock_data.empty:
                return {}
            
            ratios = {}
            
            # Price-based calculations
            current_price = stock_data['Close'].iloc[-1]
            ratios['current_price'] = current_price
            
            # 52-week high/low
            ratios['52_week_high'] = stock_data['High'].max()
            ratios['52_week_low'] = stock_data['Low'].min()
            ratios['distance_from_52w_high'] = ((current_price - ratios['52_week_high']) / ratios['52_week_high']) * 100
            ratios['distance_from_52w_low'] = ((current_price - ratios['52_week_low']) / ratios['52_week_low']) * 100
            
            # Moving averages
            if len(stock_data) >= 20:
                ratios['ma_20'] = stock_data['Close'].rolling(window=20).mean().iloc[-1]
                ratios['price_vs_ma20'] = ((current_price - ratios['ma_20']) / ratios['ma_20']) * 100
            
            if len(stock_data) >= 50:
                ratios['ma_50'] = stock_data['Close'].rolling(window=50).mean().iloc[-1]
                ratios['price_vs_ma50'] = ((current_price - ratios['ma_50']) / ratios['ma_50']) * 100
            
            if len(stock_data) >= 200:
                ratios['ma_200'] = stock_data['Close'].rolling(window=200).mean().iloc[-1]
                ratios['price_vs_ma200'] = ((current_price - ratios['ma_200']) / ratios['ma_200']) * 100
            
            # Volatility measures
            if len(stock_data) >= 30:
                returns = stock_data['Close'].pct_change().dropna()
                ratios['volatility_30d'] = returns.tail(30).std() * np.sqrt(252) * 100  # Annualized volatility
            
            # Volume analysis
            ratios['avg_volume_30d'] = stock_data['Volume'].tail(30).mean()
            ratios['current_volume'] = stock_data['Volume'].iloc[-1]
            ratios['volume_ratio'] = ratios['current_volume'] / ratios['avg_volume_30d'] if ratios['avg_volume_30d'] > 0 else 0
            
            # Price change analysis
            if len(stock_data) >= 2:
                yesterday_price = stock_data['Close'].iloc[-2]
                ratios['daily_change'] = current_price - yesterday_price
                ratios['daily_change_pct'] = (ratios['daily_change'] / yesterday_price) * 100 if yesterday_price > 0 else 0
            
            # Weekly change
            if len(stock_data) >= 7:
                week_ago_price = stock_data['Close'].iloc[-7]
                ratios['weekly_change'] = current_price - week_ago_price
                ratios['weekly_change_pct'] = (ratios['weekly_change'] / week_ago_price) * 100 if week_ago_price > 0 else 0
            
            # Monthly change
            if len(stock_data) >= 30:
                month_ago_price = stock_data['Close'].iloc[-30]
                ratios['monthly_change'] = current_price - month_ago_price
                ratios['monthly_change_pct'] = (ratios['monthly_change'] / month_ago_price) * 100 if month_ago_price > 0 else 0
            
            return ratios
            
        except Exception as e:
            st.error(f"Error calculating financial ratios: {str(e)}")
            return {}
    
    def get_dividend_info(self):
        """
        Fetch dividend information for the stock
        
        Returns:
            pd.DataFrame: Dividend history data
        """
        try:
            if not self.ticker:
                self.ticker = yf.Ticker(self.symbol)
            
            dividends = self.ticker.dividends
            
            if dividends.empty:
                return pd.DataFrame()
            
            # Get recent dividends (last 5 years)
            recent_dividends = dividends.tail(20)  # Last 20 dividend payments
            
            return recent_dividends
            
        except Exception as e:
            st.error(f"Error fetching dividend info for {self.symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_splits_info(self):
        """
        Fetch stock split information
        
        Returns:
            pd.DataFrame: Stock split history
        """
        try:
            if not self.ticker:
                self.ticker = yf.Ticker(self.symbol)
            
            splits = self.ticker.splits
            
            if splits.empty:
                return pd.DataFrame()
            
            return splits
            
        except Exception as e:
            st.error(f"Error fetching splits info for {self.symbol}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_rsi(self, stock_data, window=14):
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            stock_data (pd.DataFrame): Historical stock data
            window (int): RSI calculation window (default: 14)
            
        Returns:
            pd.Series: RSI values
        """
        try:
            if stock_data is None or len(stock_data) < window + 1:
                return pd.Series()
            
            delta = stock_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            st.error(f"Error calculating RSI: {str(e)}")
            return pd.Series()
    
    def calculate_bollinger_bands(self, stock_data, window=20, num_std=2):
        """
        Calculate Bollinger Bands
        
        Args:
            stock_data (pd.DataFrame): Historical stock data
            window (int): Moving average window (default: 20)
            num_std (int): Number of standard deviations (default: 2)
            
        Returns:
            dict: Dictionary containing upper, lower bands and moving average
        """
        try:
            if stock_data is None or len(stock_data) < window:
                return {}
            
            close_prices = stock_data['Close']
            moving_avg = close_prices.rolling(window=window).mean()
            std_dev = close_prices.rolling(window=window).std()
            
            upper_band = moving_avg + (std_dev * num_std)
            lower_band = moving_avg - (std_dev * num_std)
            
            return {
                'upper_band': upper_band,
                'lower_band': lower_band,
                'moving_average': moving_avg,
                'current_position': 'Above Upper' if close_prices.iloc[-1] > upper_band.iloc[-1] 
                                  else 'Below Lower' if close_prices.iloc[-1] < lower_band.iloc[-1] 
                                  else 'Within Bands'
            }
            
        except Exception as e:
            st.error(f"Error calculating Bollinger Bands: {str(e)}")
            return {}
