import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Dashboard",
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
    .positive { color: #00cc96; }
    .negative { color: #ef553b; }
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
                return None
                
            return {
                'historical': hist_data,
                'symbol': symbol,
                'current_price': hist_data['Close'].iloc[-1],
                'previous_close': hist_data['Close'].iloc[-2] if len(hist_data) > 1 else hist_data['Close'].iloc[-1]
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
            return macd, macd_signal
        except:
            empty_series = pd.Series([0] * len(prices), index=prices.index)
            return empty_series, empty_series

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
        if df.empty or len(df) < 30:
            return {}, df
        
        try:
            # Calculate returns and volatility first
            df['returns'] = df['Close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            
            # RSI
            df['rsi'] = self.calculate_rsi(df['Close'])
            
            # MACD
            macd, macd_signal = self.calculate_macd(df['Close'])
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['Close'])
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            
            # Moving averages
            df['sma_20'] = df['Close'].rolling(window=20).mean()
            df['sma_50'] = df['Close'].rolling(window=50).mean()
            
            latest = df.iloc[-1]
            
            # Handle NaN values safely
            rsi_value = 50 if pd.isna(latest['rsi']) else latest['rsi']
            volatility_value = 0.3 if pd.isna(latest['volatility']) else latest['volatility']
            
            signals = {
                'rsi': rsi_value,
                'rsi_signal': 'OVERSOLD' if rsi_value < 30 else 'OVERBOUGHT' if rsi_value > 70 else 'NEUTRAL',
                'macd_signal': 'BULLISH' if latest['macd'] > latest['macd_signal'] else 'BEARISH',
                'trend': 'BULLISH' if latest['sma_20'] > latest['sma_50'] else 'BEARISH' if pd.notna(latest['sma_20']) and pd.notna(latest['sma_50']) else 'NEUTRAL',
                'bb_signal': 'OVERBOUGHT' if latest['Close'] > latest['bb_upper'] else 'OVERSOLD' if latest['Close'] < latest['bb_lower'] else 'NEUTRAL',
                'volatility': volatility_value,
                'current_price': latest['Close'],
                'price_change': ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100) if len(df) > 1 else 0
            }
            
            return signals, df
            
        except Exception as e:
            st.error(f"Error calculating indicators: {str(e)}")
            # Return basic data without indicators
            basic_signals = {
                'rsi': 50,
                'rsi_signal': 'NEUTRAL',
                'macd_signal': 'NEUTRAL',
                'trend': 'NEUTRAL',
                'bb_signal': 'NEUTRAL',
                'volatility': 0.3,
                'current_price': df['Close'].iloc[-1] if not df.empty else 0,
                'price_change': 0
            }
            return basic_signals, df

    def analyze_stock(self, symbol, technical_data, price_data):
        """Analyze stock and provide recommendations"""
        risk_score = min(technical_data['volatility'] * 15, 10)
        
        # Calculate return potential based on momentum and trend
        trend_score = 1 if technical_data['trend'] == 'BULLISH' else -1 if technical_data['trend'] == 'BEARISH' else 0
        rsi_score = 0.5 if technical_data['rsi_signal'] == 'NEUTRAL' else 0.2 if technical_data['rsi_signal'] == 'OVERSOLD' else -0.2
        macd_score = 0.3 if technical_data['macd_signal'] == 'BULLISH' else -0.3
        
        return_potential = max(min((trend_score + rsi_score + macd_score) * 20, 25), -10)
        
        analysis = f"""
**Analysis for {symbol}**

**Current Price:** ${technical_data['current_price']:.2f}
**Price Change:** {technical_data['price_change']:+.2f}%

**Technical Indicators:**
- RSI: {technical_data['rsi']:.1f} ({technical_data['rsi_signal']})
- MACD: {technical_data['macd_signal']}
- Trend: {technical_data['trend']}
- Volatility: {technical_data['volatility']:.1%}

**Risk Assessment:** {risk_score:.1f}/10
**Return Potential:** {return_potential:.1f}%

**Recommendation:** {'Consider for investment' if return_potential > 5 and risk_score < 7 else 'Monitor for entry points' if return_potential > 0 else 'Wait for better conditions'}
"""
        return {
            'analysis': analysis,
            'risk_score': risk_score,
            'return_potential': return_potential,
            'allocation': max(min(return_potential / max(risk_score, 1) * 8, 12), 2)
        }

    def optimize_portfolio(self, stocks_analysis):
        """Simple portfolio optimization"""
        if len(stocks_analysis) < 2:
            return None
        
        weights = {}
        total_score = 0
        
        for symbol, data in stocks_analysis.items():
            analysis = data.get('analysis', {})
            risk_score = analysis.get('risk_score', 5)
            return_potential = analysis.get('return_potential', 5)
            
            # Risk-adjusted score (Sharpe-like)
            score = (return_potential + 5) / max(risk_score, 1)  # Add 5 to avoid negative scores
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
            go.Scatter(
                x=df.index, 
                y=df['Close'], 
                name='Close Price',
                line=dict(color='#1f77b4')
            ),
            row=1, col=1
        )
        
        # Moving averages if available
        if 'sma_20' in df.columns and not df['sma_20'].isna().all():
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df['sma_20'], 
                    name='SMA 20',
                    line=dict(color='orange', dash='dash')
                ),
                row=1, col=1
            )
        
        # Volume
        colors = ['red' if row['Open'] > row['Close'] else 'green' 
                 for _, row in df.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=df.index, 
                y=df['Volume'], 
                name='Volume',
                marker_color=colors, 
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=400, 
            showlegend=True, 
            template="plotly_white",
            margin=dict(t=50, l=50, r=50, b=50)
        )
        return fig
    except Exception as e:
        # Return empty figure if error
        return go.Figure()

def create_technical_chart(df, symbol):
    """Create technical indicators chart"""
    try:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} RSI', f'{symbol} MACD')
        )
        
        # RSI
        if 'rsi' in df.columns and not df['rsi'].isna().all():
            fig.add_trace(
                go.Scatter(x=df.index, y=df['rsi'], name='RSI',
                          line=dict(color='purple')),
                row=1, col=1
            )
            # Overbought/Oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        
        # MACD
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            if not df['macd'].isna().all():
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['macd'], name='MACD',
                              line=dict(color='blue')),
                    row=2, col=1
                )
            if not df['macd_signal'].isna().all():
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['macd_signal'], name='Signal',
                              line=dict(color='red')),
                    row=2, col=1
                )
        
        fig.update_layout(height=400, showlegend=True, template="plotly_white")
        return fig
    except Exception as e:
        return go.Figure()

def main():
    st.markdown('<h1 class="main-header">📈 Stock Analysis Dashboard</h1>', 
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
    symbols_input = st.sidebar.text_area(
        "Stock Symbols (comma separated)", 
        value=default_symbols,
        height=80
    )
    
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    
    # Analysis period
    period = st.sidebar.selectbox(
        "Analysis Period", 
        ["1mo", "3mo", "6mo", "1y"], 
        index=2
    )
    
    # Analyze button
    if st.sidebar.button("Analyze Stocks", use_container_width=True):
        with st.spinner("Analyzing stocks..."):
            st.session_state.analysis_results = {}
            
            progress_bar = st.progress(0)
            for i, symbol in enumerate(symbols):
                try:
                    # Get stock data
                    stock_data = st.session_state.analyzer.get_stock_data(symbol, period)
                    if stock_data is None:
                        st.warning(f"Could not fetch data for {symbol}")
                        continue
                    
                    # Calculate technical indicators
                    technical_signals, df_with_indicators = st.session_state.analyzer.calculate_technical_indicators(
                        stock_data['historical']
                    )
                    
                    # Analyze stock
                    analysis = st.session_state.analyzer.analyze_stock(
                        symbol, technical_signals, stock_data
                    )
                    
                    # Store results
                    st.session_state.analysis_results[symbol] = {
                        'technical': technical_signals,
                        'analysis': analysis,
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
        <h2>Welcome to the Stock Analysis Dashboard! 📊</h2>
        <p>Get <strong>AI-powered stock analysis</strong> with technical indicators and portfolio optimization.</p>
        
        <div style='display: inline-block; text-align: left; margin: 2rem 0;'>
            <h4>🎯 Features:</h4>
            <ul>
                <li>Real-time stock data from Yahoo Finance</li>
                <li>Technical analysis (RSI, MACD, Bollinger Bands)</li>
                <li>Risk assessment and return potential</li>
                <li>Portfolio optimization suggestions</li>
                <li>Interactive charts and visualizations</li>
            </ul>
        </div>
        
        <p><strong>How to use:</strong></p>
        <ol style='display: inline-block; text-align: left;'>
            <li>Enter stock symbols in the sidebar (comma separated)</li>
            <li>Select analysis period</li>
            <li>Click "Analyze Stocks"</li>
            <li>View detailed analysis and recommendations</li>
        </ol>
        
        <br>
        <p><em>Try analyzing: AAPL, MSFT, GOOG, TSLA, AMZN, NFLX</em></p>
    </div>
    """, unsafe_allow_html=True)

def display_results(analysis_results, symbols):
    """Display analysis results"""
    
    # Portfolio Optimization
    if len(analysis_results) >= 2:
        st.header("💰 Portfolio Optimization")
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
                if portfolio_weights:
                    fig = go.Figure(data=[go.Pie(
                        labels=list(portfolio_weights.keys()),
                        values=list(portfolio_weights.values()),
                        hole=.3
                    )])
                    fig.update_layout(
                        title="Portfolio Allocation",
                        height=300,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Individual Stock Analysis
    st.header("🔍 Stock Analysis")
    
    for symbol in symbols:
        if symbol not in analysis_results:
            continue
            
        data = analysis_results[symbol]
        technical = data['technical']
        analysis = data['analysis']
        
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"### {symbol}")
                
                # Price and change
                price_change = technical.get('price_change', 0)
                change_color = "positive" if price_change >= 0 else "negative"
                
                st.metric(
                    label="Current Price",
                    value=f"${technical['current_price']:.2f}",
                    delta=f"{price_change:+.2f}%"
                )
                
                # Key metrics
                st.metric("RSI", f"{technical['rsi']:.1f}", technical['rsi_signal'])
                st.metric("Trend", technical['trend'])
                st.metric("Volatility", f"{technical['volatility']:.1%}")
                
            with col2:
                # Charts
                price_chart = create_price_chart(data['df_with_indicators'], symbol)
                st.plotly_chart(price_chart, use_container_width=True)
                
                # Analysis
                st.subheader("Analysis & Recommendation")
                st.markdown(analysis['analysis'])
            
            st.markdown("---")

if __name__ == "__main__":
    main()
