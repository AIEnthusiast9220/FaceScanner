import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from stock_analyzer import StockAnalyzer
from predictor import StockPredictor

# Configure the page
st.set_page_config(
    page_title="Indian Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("📈 Indian Stock Analysis Dashboard")
    st.markdown("**Analyze NSE and BSE listed stocks with real-time data, technical indicators, and AI-powered predictions**")
    st.markdown("---")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Configuration")
        
        # Stock symbol input
        symbol = st.text_input(
            "Enter Stock Symbol",
            value="RELIANCE.NS",
            placeholder="e.g., RELIANCE.NS, TCS.NS, INFY.NS",
            help="Enter NSE stock symbol with .NS suffix (e.g., RELIANCE.NS)"
        ).upper()
        
        # Auto-append .NS if not present and doesn't already have an exchange suffix
        if symbol and '.' not in symbol:
            symbol = f"{symbol}.NS"
        
        # Popular Indian stocks quick select
        st.markdown("**Popular Indian Stocks:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏭 RELIANCE"):
                symbol = "RELIANCE.NS"
            if st.button("💻 TCS"):
                symbol = "TCS.NS"
            if st.button("🏦 HDFCBANK"):
                symbol = "HDFCBANK.NS"
        with col2:
            if st.button("💻 INFY"):
                symbol = "INFY.NS"
            if st.button("🏦 ICICIBANK"):
                symbol = "ICICIBANK.NS"
            if st.button("⚡ ADANIGREEN"):
                symbol = "ADANIGREEN.NS"
        
        st.markdown("---")
        st.markdown("**Exchange Info:**")
        st.markdown("• **NSE**: Add .NS suffix (e.g., RELIANCE.NS)")
        st.markdown("• **BSE**: Add .BO suffix (e.g., RELIANCE.BO)")
        st.markdown("• Auto-detection: Enter symbol without suffix for NSE")

    
    # Time period selection on main page
    st.markdown("### 📅 Select Analysis Period")
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo", 
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y"
    }
    
    # Create horizontal radio buttons
    selected_period = st.radio(
        "Choose time period:",
        options=list(period_options.keys()),
        index=3,  # Default to "1 Year"
        horizontal=True,
        label_visibility="collapsed"
    )
    
    period = period_options[selected_period]
    
    st.markdown("---")
    
    # Main content area
    if symbol:
        try:
            # Initialize analyzer and predictor
            analyzer = StockAnalyzer(symbol)
            predictor = StockPredictor()
            
            # Validate symbol and fetch data
            with st.spinner(f"Fetching data for {symbol}..."):
                if not analyzer.validate_symbol():
                    st.error(f"❌ Invalid stock symbol: {symbol}. Please enter a valid ticker symbol.")
                    return
                
                # Get stock data
                stock_data = analyzer.get_stock_data(period)
                if stock_data is None or stock_data.empty:
                    st.error(f"❌ No data available for {symbol}. Please try a different symbol.")
                    return
                
                # Get company info
                company_info = analyzer.get_company_info()
                
            # Display company header
            if company_info:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(f"{company_info.get('longName', symbol)} ({symbol})")
                    if 'sector' in company_info and 'industry' in company_info:
                        st.caption(f"{company_info['sector']} • {company_info['industry']}")
                with col2:
                    current_price = stock_data['Close'].iloc[-1]
                    prev_price = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
                    price_change = current_price - prev_price
                    price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
                    
                    st.metric(
                        "Current Price",
                        f"₹{current_price:.2f}",
                        f"₹{price_change:+.2f} ({price_change_pct:+.2f}%)"
                    )
            
            st.markdown("---")
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Charts", "🔮 Predictions", "📋 Detailed Metrics"])
            
            with tab1:
                # Key metrics overview
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    volume = stock_data['Volume'].iloc[-1]
                    avg_volume = stock_data['Volume'].mean()
                    st.metric("Volume", f"{volume:,.0f}", f"Avg: {avg_volume:,.0f}")
                
                with col2:
                    high_52w = stock_data['High'].max()
                    low_52w = stock_data['Low'].min()
                    st.metric("52W High", f"₹{high_52w:.2f}")
                    st.metric("52W Low", f"₹{low_52w:.2f}")
                
                with col3:
                    if company_info and 'marketCap' in company_info:
                        market_cap = company_info['marketCap']
                        if market_cap > 1e12:
                            market_cap_str = f"₹{market_cap/1e12:.2f}T"
                        elif market_cap > 1e9:
                            market_cap_str = f"₹{market_cap/1e9:.2f}B"
                        else:
                            market_cap_str = f"₹{market_cap/1e6:.2f}M"
                        st.metric("Market Cap", market_cap_str)
                    
                    if company_info and 'trailingPE' in company_info:
                        pe_ratio = company_info['trailingPE']
                        st.metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
                
                with col4:
                    if company_info and 'dividendYield' in company_info:
                        div_yield = company_info['dividendYield']
                        if div_yield:
                            st.metric("Dividend Yield", f"{div_yield*100:.2f}%")
                        else:
                            st.metric("Dividend Yield", "N/A")
                    
                    if company_info and 'beta' in company_info:
                        beta = company_info['beta']
                        st.metric("Beta", f"{beta:.2f}" if beta else "N/A")
                
                # Recent data table
                st.subheader("Recent Price Data")
                recent_data = stock_data.tail(10).copy()
                recent_data.index = recent_data.index.strftime('%Y-%m-%d')
                recent_data = recent_data[['Open', 'High', 'Low', 'Close', 'Volume']].round(2)
                recent_data['Volume'] = recent_data['Volume'].apply(lambda x: f"{x:,.0f}")
                st.dataframe(recent_data, use_container_width=True)
            
            with tab2:
                # Price chart
                st.subheader("Price Chart")
                
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3],
                    subplot_titles=('Price', 'Volume')
                )
                
                # Candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=stock_data.index,
                        open=stock_data['Open'],
                        high=stock_data['High'],
                        low=stock_data['Low'],
                        close=stock_data['Close'],
                        name="Price"
                    ),
                    row=1, col=1
                )
                
                # Volume chart
                fig.add_trace(
                    go.Bar(
                        x=stock_data.index,
                        y=stock_data['Volume'],
                        name="Volume",
                        marker_color='rgba(158,202,225,0.6)'
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    title=f"{symbol} Price and Volume",
                    xaxis_rangeslider_visible=False,
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Moving averages
                st.subheader("Moving Averages")
                
                # Calculate moving averages
                stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
                stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
                
                fig_ma = go.Figure()
                
                fig_ma.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue')
                ))
                
                fig_ma.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['MA20'],
                    mode='lines',
                    name='20-day MA',
                    line=dict(color='orange')
                ))
                
                fig_ma.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['MA50'],
                    mode='lines',
                    name='50-day MA',
                    line=dict(color='red')
                ))
                
                fig_ma.update_layout(
                    title=f"{symbol} Price with Moving Averages",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=400
                )
                
                st.plotly_chart(fig_ma, use_container_width=True)
            
            with tab3:
                st.subheader("🔮 Price Predictions & Trading Targets")
                
                with st.spinner("Generating predictions..."):
                    # Get predictions
                    predictions = predictor.predict_prices(stock_data)
                    targets = predictor.calculate_targets(stock_data, predictions)
                
                if predictions and targets:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📅 Short-term Outlook (7 days)")
                        
                        short_pred = predictions['short_term']
                        short_targets = targets['short_term']
                        
                        st.metric(
                            "Predicted Price",
                            f"₹{short_pred['price']:.2f}",
                            f"{short_pred['change_pct']:+.2f}%"
                        )
                        
                        st.markdown("**Trading Targets:**")
                        st.success(f"🎯 **Buy Target:** ₹{short_targets['buy']:.2f}")
                        st.error(f"🎯 **Sell Target:** ₹{short_targets['sell']:.2f}")
                        st.info(f"📊 **Confidence:** {short_pred['confidence']:.1f}%")
                    
                    with col2:
                        st.markdown("### 📆 Long-term Outlook (30 days)")
                        
                        long_pred = predictions['long_term']
                        long_targets = targets['long_term']
                        
                        st.metric(
                            "Predicted Price",
                            f"₹{long_pred['price']:.2f}",
                            f"{long_pred['change_pct']:+.2f}%"
                        )
                        
                        st.markdown("**Trading Targets:**")
                        st.success(f"🎯 **Buy Target:** ₹{long_targets['buy']:.2f}")
                        st.error(f"🎯 **Sell Target:** ₹{long_targets['sell']:.2f}")
                        st.info(f"📊 **Confidence:** {long_pred['confidence']:.1f}%")
                    
                    # Prediction visualization
                    st.markdown("### 📈 Prediction Visualization")
                    
                    # Create prediction chart
                    fig_pred = go.Figure()
                    
                    # Historical data
                    recent_data = stock_data.tail(30)
                    fig_pred.add_trace(go.Scatter(
                        x=recent_data.index,
                        y=recent_data['Close'],
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='blue')
                    ))
                    
                    # Prediction points
                    current_price = stock_data['Close'].iloc[-1]
                    last_date = stock_data.index[-1]
                    
                    # Short-term prediction
                    short_date = last_date + timedelta(days=7)
                    fig_pred.add_trace(go.Scatter(
                        x=[last_date, short_date],
                        y=[current_price, short_pred['price']],
                        mode='lines+markers',
                        name='Short-term Prediction',
                        line=dict(color='orange', dash='dash'),
                        marker=dict(size=8)
                    ))
                    
                    # Long-term prediction
                    long_date = last_date + timedelta(days=30)
                    fig_pred.add_trace(go.Scatter(
                        x=[last_date, long_date],
                        y=[current_price, long_pred['price']],
                        mode='lines+markers',
                        name='Long-term Prediction',
                        line=dict(color='red', dash='dash'),
                        marker=dict(size=8)
                    ))
                    
                    fig_pred.update_layout(
                        title=f"{symbol} Price Predictions",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Disclaimer
                    st.warning("""
                    ⚠️ **Disclaimer:** These predictions are based on simple statistical models and should not be considered as financial advice. 
                    Always do your own research and consider consulting with a financial advisor before making investment decisions.
                    """)
                else:
                    st.error("Unable to generate predictions. Please try again with a different stock symbol.")
            
            with tab4:
                st.subheader("📋 Detailed Financial Metrics")
                
                if company_info:
                    # Create detailed metrics table
                    metrics_data = {}
                    
                    # Valuation metrics
                    valuation_metrics = {
                        'Market Cap': company_info.get('marketCap'),
                        'Enterprise Value': company_info.get('enterpriseValue'),
                        'Trailing P/E': company_info.get('trailingPE'),
                        'Forward P/E': company_info.get('forwardPE'),
                        'Price to Sales': company_info.get('priceToSalesTrailing12Months'),
                        'Price to Book': company_info.get('priceToBook'),
                        'Enterprise to Revenue': company_info.get('enterpriseToRevenue'),
                        'Enterprise to EBITDA': company_info.get('enterpriseToEbitda')
                    }
                    
                    # Financial metrics
                    financial_metrics = {
                        'Total Revenue': company_info.get('totalRevenue'),
                        'Revenue Per Share': company_info.get('revenuePerShare'),
                        'Total Cash': company_info.get('totalCash'),
                        'Total Debt': company_info.get('totalDebt'),
                        'Current Ratio': company_info.get('currentRatio'),
                        'Return on Assets': company_info.get('returnOnAssets'),
                        'Return on Equity': company_info.get('returnOnEquity'),
                        'Gross Margins': company_info.get('grossMargins'),
                        'Operating Margins': company_info.get('operatingMargins'),
                        'Profit Margins': company_info.get('profitMargins')
                    }
                    
                    # Stock metrics
                    stock_metrics = {
                        'Beta': company_info.get('beta'),
                        'Shares Outstanding': company_info.get('sharesOutstanding'),
                        'Float Shares': company_info.get('floatShares'),
                        'Held by Insiders': company_info.get('heldPercentInsiders'),
                        'Held by Institutions': company_info.get('heldPercentInstitutions'),
                        'Short Ratio': company_info.get('shortRatio'),
                        'Short % of Shares': company_info.get('shortPercentOfFloat'),
                        'Book Value': company_info.get('bookValue'),
                        'Dividend Rate': company_info.get('dividendRate'),
                        'Dividend Yield': company_info.get('dividendYield')
                    }
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Valuation Metrics**")
                        valuation_df = pd.DataFrame(
                            [(k, format_metric(v)) for k, v in valuation_metrics.items() if v is not None],
                            columns=['Metric', 'Value']
                        )
                        if not valuation_df.empty:
                            st.dataframe(valuation_df, hide_index=True, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Financial Metrics**")
                        financial_df = pd.DataFrame(
                            [(k, format_metric(v)) for k, v in financial_metrics.items() if v is not None],
                            columns=['Metric', 'Value']
                        )
                        if not financial_df.empty:
                            st.dataframe(financial_df, hide_index=True, use_container_width=True)
                    
                    with col3:
                        st.markdown("**Stock Metrics**")
                        stock_df = pd.DataFrame(
                            [(k, format_metric(v)) for k, v in stock_metrics.items() if v is not None],
                            columns=['Metric', 'Value']
                        )
                        if not stock_df.empty:
                            st.dataframe(stock_df, hide_index=True, use_container_width=True)
                    
                    # Company description
                    if 'longBusinessSummary' in company_info:
                        st.markdown("### Company Overview")
                        st.write(company_info['longBusinessSummary'])
                else:
                    st.warning("Detailed company information not available for this symbol.")
        
        except Exception as e:
            st.error(f"❌ An error occurred while analyzing {symbol}: {str(e)}")
            st.error("Please check the stock symbol and try again.")
    
    elif not symbol:
        st.info("👆 Enter an Indian stock symbol in the sidebar to begin analysis.")
        
        # Add example section when no symbol is entered
        st.markdown("### 🇮🇳 Popular Indian Stocks to Try:")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**Banking & Finance**")
            st.markdown("• HDFCBANK.NS")
            st.markdown("• ICICIBANK.NS") 
            st.markdown("• SBIN.NS")
            st.markdown("• KOTAKBANK.NS")
        
        with col2:
            st.markdown("**Technology**")
            st.markdown("• TCS.NS")
            st.markdown("• INFY.NS")
            st.markdown("• WIPRO.NS")
            st.markdown("• TECHM.NS")
        
        with col3:
            st.markdown("**Energy & Materials**")
            st.markdown("• RELIANCE.NS")
            st.markdown("• ONGC.NS")
            st.markdown("• NTPC.NS")
            st.markdown("• COALINDIA.NS")
        
        with col4:
            st.markdown("**Consumer & Others**")
            st.markdown("• HINDUNILVR.NS")
            st.markdown("• ITC.NS")
            st.markdown("• BAJFINANCE.NS")
            st.markdown("• MARUTI.NS")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666666;'>"
        "📊 Indian Stock Analysis Dashboard | NSE/BSE Data via Yahoo Finance"
        "</div>",
        unsafe_allow_html=True
    )

def format_metric(value):
    """Format numerical metrics for display"""
    if value is None:
        return "N/A"
    
    if isinstance(value, (int, float)):
        if abs(value) >= 1e12:
            return f"{value/1e12:.2f}T"
        elif abs(value) >= 1e9:
            return f"{value/1e9:.2f}B"
        elif abs(value) >= 1e6:
            return f"{value/1e6:.2f}M"
        elif abs(value) >= 1000:
            return f"{value/1000:.2f}K"
        elif isinstance(value, float):
            if 0 < abs(value) < 1:
                return f"{value:.4f}"
            else:
                return f"{value:.2f}"
        else:
            return f"{value:,}"
    
    return str(value)

def format_currency(value):
    """Format currency values in rupees"""
    if value is None:
        return "N/A"
    
    if isinstance(value, (int, float)):
        if abs(value) >= 1e7:  # 1 crore
            return f"₹{value/1e7:.2f}Cr"
        elif abs(value) >= 1e5:  # 1 lakh
            return f"₹{value/1e5:.2f}L"
        elif abs(value) >= 1e3:
            return f"₹{value/1e3:.2f}K"
        else:
            return f"₹{value:.2f}"
    
    return f"₹{value}"

if __name__ == "__main__":
    main()
