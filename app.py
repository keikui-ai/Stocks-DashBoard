import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import json

# Page configuration
st.set_page_config(
    page_title="AI Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stock-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class StockAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.02
        
    def get_stock_data(self, symbol, period="6mo"):
        """Get stock data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period=period)
            
            if hist_data.empty:
                st.error(f"No data found for {symbol}")
                return None
                
            return {
                'historical': hist_data,
                'symbol': symbol
            }
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices), index=prices.index)

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        try:
            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=signal, adjust=False).mean()
            macd_hist = macd - macd_signal
            return macd, macd_signal, macd_hist
        except:
            empty_series = pd.Series([0] * len(prices), index=prices.index)
            return empty_series, empty_series, empty_series

    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        try:
            rolling_mean = prices.rolling(window=window).mean()
            rolling_std = prices.rolling(window=window).std()
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            return upper_band, rolling_mean, lower_band
        except:
            empty_series = pd.Series([0] * len(prices), index=prices.index)
            return empty_series, empty_series, empty_series

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        if df.empty or len(df) < 50:
            return {}, df
        
        try:
            # RSI
            df['rsi'] = self.calculate_rsi(df['Close'])
            
            # MACD
            macd, macd_signal, macd_hist = self.calculate_macd(df['Close'])
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['Close'])
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            
            # Moving averages
            df['sma_20'] = df['Close'].rolling(window=20).mean()
            df['sma_50'] = df['Close'].rolling(window=50).mean()
            
            # Volatility
            df['volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            
            # Momentum
            if len(df) > 21:
                df['momentum_1m'] = df['Close'].pct_change(21)
            else:
                df['momentum_1m'] = 0
                
            latest = df.iloc[-1]
            
            # Handle NaN values
            rsi_value = latest['rsi'] if not pd.isna(latest['rsi']) else 50
            volatility_value = latest['volatility'] if not pd.isna(latest['volatility']) else 0.3
            momentum_value = latest['momentum_1m'] if not pd.isna(latest['momentum_1m']) else 0
            
            signals = {
                'rsi': rsi_value,
                'rsi_signal': 'OVERSOLD' if rsi_value < 30 else 'OVERBOUGHT' if rsi_value > 70 else 'NEUTRAL',
                'macd_signal': 'BULLISH' if latest['macd'] > latest['macd_signal'] else 'BEARISH',
                'trend': 'BULLISH' if latest['sma_20'] > latest['sma_50'] else 'BEARISH',
                'bb_signal': 'OVERBOUGHT' if latest['Close'] > latest['bb_upper'] else 'OVERSOLD' if latest['Close'] < latest['bb_lower'] else 'NEUTRAL',
                'volatility': volatility_value,
                'momentum': momentum_value
            }
            
            return signals, df
            
        except Exception as e:
            st.error(f"Error calculating indicators: {str(e)}")
            return {}, df

    def analyze_with_ai(self, symbol, technical_data):
        """AI analysis fallback without API"""
        risk_score = min(technical_data['volatility'] * 10, 10)
        return_potential = abs(technical_data['momentum'] * 100)
        
        analysis = f"""
**AI Analysis for {symbol}:**

**Risk Assessment:** {risk_score:.1f}/10
**Return Potential:** {return_potential:.1f}%
**Technical Outlook:** {technical_data['trend']}

**Key Indicators:**
- RSI: {technical_data['rsi']:.1f} ({technical_data['rsi_signal']})
- MACD: {technical_data['macd_signal']}
- Volatility: {technical_data['volatility']:.1%}

**Recommendation:** {'Consider for portfolio' if return_potential > 5 and risk_score < 7 else 'Monitor for now'}
"""
        return {
            'analysis': analysis,
            'risk_score': risk_score,
            'return_potential': return_potential,
            'allocation': max(min(return_potential / max(risk_score, 1) * 10, 15), 2)
        }

    def optimize_portfolio(self, stocks_analysis):
        """Simple portfolio optimization"""
        if len(stocks_analysis) < 2:
            return None
        
        weights = {}
        total_score = 0
        
        for symbol, data in stocks_analysis.items():
            ai_analysis = data.get('ai_analysis', {})
            risk_score = ai_analysis.get('risk_score', 5)
            return_potential = ai_analysis.get('return_potential', 10)
            
            # Risk-adjusted score
            score = return_potential / max(risk_score, 1)
            weights[symbol] = score
            total_score += score
        
        if total_score > 0:
            weights = {k: (v / total_score) * 100 for k, v in weights.items()}
        else:
            equal_weight = 100 / len(stocks_analysis)
            weights = {k: equal_weight for k in stocks_analysis.keys()}
        
        return weights

def create_price_chart(df, symbol):
    """Create interactive price chart"""
    try:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Price Chart', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Price line
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Close'], name='Close Price',
                      line=dict(color='#1f77b4')),
            row=1, col=1
        )
        
        # Moving averages if available
        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20',
                          line=dict(color='orange', dash='dash')),
                row=1, col=1
            )
        
        # Volume
        colors = ['red' if row['Open'] > row['Close'] else 'green' 
                 for _, row in df.iterrows()]
        
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume',
                   marker_color=colors, opacity=0.7),
            row=2, col=1
        )
        
        fig.update_layout(height=400, showlegend=True, template="plotly_white")
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return go.Figure()

def main():
    st.markdown('<h1 class="main-header">🤖 Stock Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = StockAnalyzer()
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    
    # Sidebar
    st.sidebar.header("Stock Selection")
    
    # Stock selection
    default_symbols = "AAPL, MSFT, GOOG, TSLA"
    symbols_input = st.sidebar.text_area("Stock Symbols (comma separated)", 
                                        value=default_symbols,
                                        height=100)
    
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    
    # Analysis period
    period = st.sidebar.selectbox("Analysis Period", 
                                 ["3mo", "6mo", "1y"], 
                                 index=1)
    
    # Analyze button
    if st.sidebar.button("🚀 Analyze Stocks", use_container_width=True):
        with st.spinner("Analyzing stocks... This may take a few moments."):
            st.session_state.analysis_results = {}
            
            progress_bar = st.progress(0)
            for i, symbol in enumerate(symbols):
                try:
                    st.write(f"📊 Analyzing {symbol}...")
                    
                    # Get stock data
                    stock_data = st.session_state.analyzer.get_stock_data(symbol, period)
                    if stock_data is None:
                        continue
                    
                    # Calculate technical indicators
                    technical_signals, df_with_indicators = st.session_state.analyzer.calculate_technical_indicators(
                        stock_data['historical']
                    )
                    
                    if not technical_signals:
                        st.warning(f"Not enough data for technical analysis of {symbol}")
                        continue
                    
                    # AI Analysis
                    ai_analysis = st.session_state.analyzer.analyze_with_ai(
                        symbol, technical_signals
                    )
                    
                    # Store results
                    st.session_state.analysis_results[symbol] = {
                        'technical': technical_signals,
                        'ai_analysis': ai_analysis,
                        'historical': stock_data['historical'],
                        'df_with_indicators': df_with_indicators
                    }
                    
                    progress_bar.progress((i + 1) / len(symbols))
                    
                except Exception as e:
                    st.error(f"Error analyzing {symbol}: {str(e)}")
                    continue
    
    # Main content
    if st.session_state.analysis_results:
        display_results(st.session_state.analysis_results, symbols)
    else:
        display_welcome()

def display_welcome():
    """Display welcome message"""
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h2>Welcome to the Stock Analysis Dashboard! 🎯</h2>
        <p>Analyze stocks for <strong>risk-adjusted returns</strong> using:</p>
        <ul style='display: inline-block; text-align: left;'>
            <li>📊 <strong>Real-time data</strong> from Yahoo Finance</li>
            <li>🔍 <strong>Technical analysis</strong> with multiple indicators</li>
            <li>🤖 <strong>AI-powered insights</strong> and recommendations</li>
            <li>💼 <strong>Portfolio optimization</strong> suggestions</li>
        </ul>
        <br><br>
        <p><strong>How to use:</strong></p>
        <ol style='display: inline-block; text-align: left;'>
            <li>Add stock symbols in the sidebar</li>
            <li>Click "Analyze Stocks"</li>
            <li>View detailed analysis and recommendations</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

def display_results(analysis_results, symbols):
    """Display analysis results"""
    
    # Portfolio Optimization
    if len(analysis_results) >= 2:
        st.header("📊 Portfolio Optimization")
        portfolio_weights = st.session_state.analyzer.optimize_portfolio(analysis_results)
        
        if portfolio_weights:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Recommended Allocation")
                for symbol, weight in sorted(portfolio_weights.items(), 
                                           key=lambda x: x[1], reverse=True):
                    st.metric(
                        label=symbol,
                        value=f"{weight:.1f}%"
                    )
            
            with col2:
                # Portfolio pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=list(portfolio_weights.keys()),
                    values=list(portfolio_weights.values()),
                    hole=.3
                )])
                fig.update_layout(title="Portfolio Allocation", height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    # Individual Stock Analysis
    st.header("🔍 Stock Analysis")
    
    for symbol in symbols:
        if symbol not in analysis_results:
            continue
            
        data = analysis_results[symbol]
        technical = data['technical']
        ai_analysis = data['ai_analysis']
        
        with st.container():
            st.markdown(f"### {symbol}")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                delta_color = "inverse" if technical['rsi_signal'] == 'OVERSOLD' else "off" if technical['rsi_signal'] == 'OVERBOUGHT' else "normal"
                st.metric("RSI", f"{technical['rsi']:.1f}", technical['rsi_signal'], delta_color=delta_color)
            
            with col2:
                st.metric("MACD", technical['macd_signal'])
            
            with col3:
                st.metric("Trend", technical['trend'])
            
            with col4:
                st.metric("Volatility", f"{(technical['volatility'] * 100):.1f}%")
            
            # Charts
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                price_chart = create_price_chart(data['df_with_indicators'], symbol)
                st.plotly_chart(price_chart, use_container_width=True)
            
            # AI Analysis
            st.subheader("🤖 Analysis & Recommendation")
            st.markdown(ai_analysis.get('analysis', 'Analysis not available'))
            
            st.markdown("---")

if __name__ == "__main__":
    main()
