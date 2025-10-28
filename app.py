import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries
import time

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
    .positive { color: #00cc96; }
    .negative { color: #ef553b; }
</style>
""", unsafe_allow_html=True)

# Cache data fetching to avoid hitting Alpha Vantage rate limits
@st.cache_data(ttl=3600)
def fetch_alpha_vantage_data(symbol, outputsize="full"):
    """Fetch daily stock data from Alpha Vantage (cached)"""
    try:
        api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
    except KeyError:
        st.error("❌ Alpha Vantage API key not found in secrets. Add it to Streamlit Secrets.")
        return None

    ts = TimeSeries(key=api_key, output_format='pandas')
    try:
        data, meta = ts.get_daily(symbol=symbol, outputsize=outputsize)
        if data.empty:
            return None
        # Rename columns to match yfinance format
        data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }, inplace=True)
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        return data
    except Exception as e:
        st.error(f"Alpha Vantage error for {symbol}: {e}")
        return None

class StockAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.02
        
    def get_stock_data(self, symbol, period="6mo"):
        """Get stock data from Alpha Vantage"""
        data = fetch_alpha_vantage_data(symbol, outputsize="full")
        if data is None or data.empty:
            st.error(f"No data found for {symbol}")
            return None

        # Slice based on period
        days_map = {
            "1mo": 21,
            "3mo": 63,
            "6mo": 126,
            "1y": 252
        }
        n_days = days_map.get(period, 126)
        sliced_data = data.tail(n_days)

        if len(sliced_data) < 2:
            st.error(f"Insufficient data for {symbol} over {period}")
            return None

        current_price = sliced_data['Close'].iloc[-1]
        previous_close = sliced_data['Close'].iloc[-2]

        return {
            'historical': sliced_data,
            'symbol': symbol,
            'current_price': current_price,
            'previous_close': previous_close
        }

    def calculate_rsi(self, prices, window=14):
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices), index=prices.index)

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
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
        if df.empty or len(df) < 30:
            return {}, df
        
        try:
            df['returns'] = df['Close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            df['rsi'] = self.calculate_rsi(df['Close'])
            macd, macd_signal = self.calculate_macd(df['Close'])
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['Close'])
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['sma_20'] = df['Close'].rolling(window=20).mean()
            df['sma_50'] = df['Close'].rolling(window=50).mean()

            latest = df.iloc[-1]
            rsi_value = latest['rsi'] if not pd.isna(latest['rsi']) else 50
            volatility_value = latest['volatility'] if not pd.isna(latest['volatility']) else 0.3
            macd_signal = 'BULLISH' if latest['macd'] > latest['macd_signal'] else 'BEARISH'
            trend = 'BULLISH' if pd.notna(latest['sma_20']) and pd.notna(latest['sma_50']) and latest['sma_20'] > latest['sma_50'] else 'BEARISH' if pd.notna(latest['sma_20']) and pd.notna(latest['sma_50']) else 'NEUTRAL'
            bb_signal = 'OVERBOUGHT' if pd.notna(latest['bb_upper']) and latest['Close'] > latest['bb_upper'] else 'OVERSOLD' if pd.notna(latest['bb_lower']) and latest['Close'] < latest['bb_lower'] else 'NEUTRAL'

            signals = {
                'rsi': rsi_value,
                'rsi_signal': 'OVERSOLD' if rsi_value < 30 else 'OVERBOUGHT' if rsi_value > 70 else 'NEUTRAL',
                'macd_signal': macd_signal,
                'trend': trend,
                'bb_signal': bb_signal,
                'volatility': volatility_value,
                'current_price': latest['Close'],
                'price_change': ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100) if len(df) > 1 else 0
            }
            return signals, df
        except Exception as e:
            st.error(f"Error calculating indicators: {str(e)}")
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

    def analyze_stock(self, symbol, technical_data):
        try:
            risk_score = min(technical_data['volatility'] * 15, 10)
            trend_score = 1 if technical_data['trend'] == 'BULLISH' else -1 if technical_data['trend'] == 'BEARISH' else 0
            rsi_score = 0.5 if technical_data['rsi_signal'] == 'NEUTRAL' else 0.8 if technical_data['rsi_signal'] == 'OVERSOLD' else -0.2
            macd_score = 0.3 if technical_data['macd_signal'] == 'BULLISH' else -0.3
            return_potential = max(min((trend_score + rsi_score + macd_score) * 25, 30), -15)
            
            analysis = f"""
**📊 Analysis for {symbol}**

**💰 Price:** ${technical_data['current_price']:.2f} ({technical_data['price_change']:+.2f}%)

**🔍 Technical Indicators:**
- **RSI:** {technical_data['rsi']:.1f} ({technical_data['rsi_signal']})
- **MACD:** {technical_data['macd_signal']}
- **Trend:** {technical_data['trend']}
- **Volatility:** {technical_data['volatility']:.1%}
- **Bollinger Bands:** {technical_data['bb_signal']}

**📈 Risk Assessment:** {risk_score:.1f}/10
**🎯 Return Potential:** {return_potential:.1f}%

**💡 Recommendation:** {'Consider for investment' if return_potential > 8 and risk_score < 6 else 'Good potential, monitor closely' if return_potential > 3 else 'Wait for better entry point'}
"""
            return {
                'analysis': analysis,
                'risk_score': risk_score,
                'return_potential': return_potential,
                'allocation': max(min(return_potential / max(risk_score, 1) * 6, 15), 2)
            }
        except Exception as e:
            st.error(f"Error in stock analysis: {str(e)}")
            return {
                'analysis': f"Basic analysis for {symbol}: Current price ${technical_data.get('current_price', 0):.2f}",
                'risk_score': 5,
                'return_potential': 5,
                'allocation': 5
            }

    def optimize_portfolio(self, stocks_analysis):
        if len(stocks_analysis) < 2:
            return None
        try:
            weights = {}
            total_score = 0
            for symbol, data in stocks_analysis.items():
                analysis = data.get('analysis', {})
                risk_score = analysis.get('risk_score', 5)
                return_potential = analysis.get('return_potential', 5)
                score = (return_potential + 5) / max(risk_score, 1)
                weights[symbol] = score
                total_score += score
            if total_score > 0:
                weights = {k: (v / total_score) * 100 for k, v in weights.items()}
            else:
                equal_weight = 100 / len(stocks_analysis)
                weights = {k: equal_weight for k in stocks_analysis.keys()}
            return weights
        except Exception as e:
            st.error(f"Error in portfolio optimization: {str(e)}")
            return None

def create_price_chart(df, symbol):
    try:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Price Chart', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='#1f77b4')), row=1, col=1)
        if 'sma_20' in df.columns and not df['sma_20'].isna().all():
            fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', line=dict(color='orange', dash='dash')), row=1, col=1)
        colors = ['red' if row['Open'] > row['Close'] else 'green' for _, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors, opacity=0.7), row=2, col=1)
        fig.update_layout(height=400, showlegend=True, template="plotly_white", margin=dict(t=50, l=50, r=50, b=50))
        return fig
    except Exception as e:
        st.error(f"Error creating price chart: {str(e)}")
        return go.Figure()

def create_technical_chart(df, symbol):
    try:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} RSI', f'{symbol} MACD')
        )
        if 'rsi' in df.columns and not df['rsi'].isna().all():
            fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple')), row=1, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            if not df['macd'].isna().all():
                fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD', line=dict(color='blue')), row=2, col=1)
            if not df['macd_signal'].isna().all():
                fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name='Signal', line=dict(color='red')), row=2, col=1)
        fig.update_layout(height=400, showlegend=True, template="plotly_white")
        return fig
    except Exception as e:
        st.error(f"Error creating technical chart: {str(e)}")
        return go.Figure()

def main():
    st.markdown('<h1 class="main-header">🤖 AI Stock Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = StockAnalyzer()
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    
    st.sidebar.header("Configuration")
    st.sidebar.subheader("Stock Selection")
    default_symbols = "AAPL, MSFT, GOOGL, TSLA, AMZN, NFLX"
    symbols_input = st.sidebar.text_area(
        "Stock Symbols (comma separated)", 
        value=default_symbols,
        height=100,
        help="Enter stock symbols separated by commas"
    )
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    
    period = st.sidebar.selectbox("Analysis Period", ["1mo", "3mo", "6mo", "1y"], index=2)
    
    if st.sidebar.button("🚀 Analyze Stocks", type="primary", use_container_width=True):
        if not symbols:
            st.warning("Please enter at least one stock symbol.")
            return

        with st.spinner("Analyzing stocks... This may take a few moments."):
            st.session_state.analysis_results = {}
            progress_bar = st.progress(0)
            for i, symbol in enumerate(symbols):
                try:
                    st.write(f"📊 Analyzing {symbol}...")
                    stock_data = st.session_state.analyzer.get_stock_data(symbol, period)
                    if stock_data is None:
                        continue
                    technical_signals, df_with_indicators = st.session_state.analyzer.calculate_technical_indicators(stock_data['historical'])
                    analysis = st.session_state.analyzer.analyze_stock(symbol, technical_signals)
                    st.session_state.analysis_results[symbol] = {
                        'technical': technical_signals,
                        'analysis': analysis,
                        'historical': stock_data['historical'],
                        'df_with_indicators': df_with_indicators
                    }
                    progress_bar.progress((i + 1) / len(symbols))
                    # Respect Alpha Vantage free tier (5 req/min): wait ~15 sec every 4 stocks
                    if (i + 1) % 4 == 0 and i + 1 < len(symbols):
                        time.sleep(15)
                except Exception as e:
                    st.error(f"Error analyzing {symbol}: {str(e)}")
                    continue
            st.success("Analysis complete! ✅")
    
    if st.session_state.analysis_results:
        display_results(st.session_state.analysis_results, symbols)
    else:
        display_welcome()

def display_welcome():
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h2>Welcome to the AI Stock Analysis Dashboard! 🎯</h2>
        <p>Get <strong>AI-powered stock analysis</strong> with technical indicators and portfolio optimization.</p>
        <div style='display: inline-block; text-align: left; margin: 2rem 0;'>
            <h4>🚀 Features:</h4>
            <ul>
                <li>📊 <strong>Real-time data</strong> from Alpha Vantage</li>
                <li>🔍 <strong>Technical analysis</strong> (RSI, MACD, Bollinger Bands)</li>
                <li>📈 <strong>Risk assessment</strong> and return potential</li>
                <li>💼 <strong>Portfolio optimization</strong> suggestions</li>
                <li>📱 <strong>Interactive charts</strong> and visualizations</li>
            </ul>
        </div>
        <p><strong>🎯 How to use:</strong></p>
        <ol style='display: inline-block; text-align: left;'>
            <li>Add your <strong>Alpha Vantage API key</strong> in Streamlit Secrets</li>
            <li>Enter stock symbols in the sidebar (comma separated)</li>
            <li>Select analysis period</li>
            <li>Click "Analyze Stocks"</li>
        </ol>
        <br>
        <p><em>💡 Try analyzing: AAPL, MSFT, GOOGL, TSLA, AMZN, NFLX</em></p>
    </div>
    """, unsafe_allow_html=True)

def display_results(analysis_results, symbols):
    if len(analysis_results) >= 2:
        st.header("💰 Portfolio Optimization")
        portfolio_weights = st.session_state.analyzer.optimize_portfolio(analysis_results)
        if portfolio_weights:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Recommended Allocation")
                for symbol, weight in sorted(portfolio_weights.items(), key=lambda x: x[1], reverse=True):
                    st.metric(label=symbol, value=f"{weight:.1f}%", help=f"Recommended portfolio allocation for {symbol}")
            with col2:
                fig = go.Figure(data=[go.Pie(labels=list(portfolio_weights.keys()), values=list(portfolio_weights.values()), hole=.3, textinfo='label+percent')])
                fig.update_layout(title="Portfolio Allocation", height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    st.header("🔍 Individual Stock Analysis")
    for symbol in symbols:
        if symbol not in analysis_results:
            continue
        data = analysis_results[symbol]
        technical = data['technical']
        analysis = data['analysis']
        with st.container():
            st.markdown(f"### {symbol}")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                price_change = technical.get('price_change', 0)
                delta_color = "normal" if price_change >= 0 else "inverse"
                st.metric("Current Price", f"${technical['current_price']:.2f}", f"{price_change:+.2f}%", delta_color=delta_color)
            with col2:
                rsi_color = "inverse" if technical['rsi_signal'] == 'OVERSOLD' else "off" if technical['rsi_signal'] == 'OVERBOUGHT' else "normal"
                st.metric("RSI", f"{technical['rsi']:.1f}", technical['rsi_signal'], delta_color=rsi_color)
            with col3:
                st.metric("Trend", technical['trend'])
            with col4:
                st.metric("Volatility", f"{technical['volatility']:.1%}")
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                price_chart = create_price_chart(data['df_with_indicators'], symbol)
                st.plotly_chart(price_chart, use_container_width=True)
            with col_chart2:
                tech_chart = create_technical_chart(data['df_with_indicators'], symbol)
                st.plotly_chart(tech_chart, use_container_width=True)
            
            st.subheader("🤖 AI Analysis & Recommendation")
            st.markdown(analysis['analysis'])
            st.markdown("---")

if __name__ == "__main__":
    main()
