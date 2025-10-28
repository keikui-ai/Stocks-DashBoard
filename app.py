# app.py
import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib
from datetime import datetime, timedelta
import json
from openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="AI Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .positive {
        color: #00cc96;
    }
    .negative {
        color: #ef553b;
    }
</style>
""", unsafe_allow_html=True)

class StockAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.02
        
    def get_stock_data(self, symbol, period="1y"):
        """Get stock data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period=period)
            info = ticker.info
            
            return {
                'historical': hist_data,
                'info': info,
                'symbol': symbol
            }
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return None

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        if df.empty:
            return {}
        
        # RSI
        df['rsi'] = talib.RSI(df['Close'], timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df['Close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
        # Moving averages
        df['sma_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['ema_12'] = talib.EMA(df['Close'], timeperiod=12)
        
        # Volatility
        df['volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # Momentum
        df['momentum_1m'] = df['Close'].pct_change(21)
        df['momentum_3m'] = df['Close'].pct_change(63)
        df['momentum_6m'] = df['Close'].pct_change(126)
        
        latest = df.iloc[-1]
        
        signals = {
            'rsi': latest['rsi'],
            'rsi_signal': 'OVERSOLD' if latest['rsi'] < 30 else 'OVERBOUGHT' if latest['rsi'] > 70 else 'NEUTRAL',
            'macd_signal': 'BULLISH' if latest['macd'] > latest['macd_signal'] else 'BEARISH',
            'trend': 'BULLISH' if latest['sma_20'] > latest['sma_50'] else 'BEARISH',
            'bb_signal': 'OVERBOUGHT' if latest['Close'] > latest['bb_upper'] else 'OVERSOLD' if latest['Close'] < latest['bb_lower'] else 'NEUTRAL',
            'volatility': latest['volatility'],
            'momentum': (latest['momentum_1m'] * 0.4 + latest['momentum_3m'] * 0.35 + latest['momentum_6m'] * 0.25)
        }
        
        return signals, df

    def analyze_with_deepseek(self, symbol, technical_data, fundamental_data):
        """Use DeepSeek for AI-powered analysis"""
        if not st.session_state.get('deepseek_api_key'):
            return self.get_fallback_analysis(technical_data)
        
        try:
            client = OpenAI(
                api_key=st.session_state.deepseek_api_key,
                base_url="https://api.deepseek.com/v1"
            )
            
            prompt = f"""
            Analyze {symbol} stock for risk-adjusted return optimization:
            
            TECHNICAL INDICATORS:
            - RSI: {technical_data['rsi']:.2f} ({technical_data['rsi_signal']})
            - MACD Signal: {technical_data['macd_signal']}
            - Trend: {technical_data['trend']}
            - Volatility: {technical_data['volatility']:.3f}
            - Momentum: {technical_data['momentum']:.3f}
            - Bollinger Band: {technical_data['bb_signal']}
            
            Please provide a concise analysis focusing on:
            1. Risk assessment (1-10 score)
            2. Return potential (percentage)
            3. Recommended allocation (0-100%)
            4. Key risk factors
            5. Risk mitigation strategies
            
            Format as JSON with: risk_score, return_potential, allocation, risks, strategies.
            """
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a senior financial analyst specializing in risk-adjusted returns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            analysis_text = response.choices[0].message.content
            return self.parse_ai_analysis(analysis_text)
            
        except Exception as e:
            st.error(f"DeepSeek API error: {e}")
            return self.get_fallback_analysis(technical_data)

    def parse_ai_analysis(self, analysis_text):
        """Parse AI analysis response"""
        try:
            if '```json' in analysis_text:
                json_str = analysis_text.split('```json')[1].split('```')[0].strip()
                return json.loads(json_str)
            else:
                return {'analysis': analysis_text}
        except:
            return {'analysis': analysis_text}

    def get_fallback_analysis(self, technical_data):
        """Fallback analysis without AI"""
        risk_score = min(technical_data['volatility'] * 10, 10)
        return_potential = technical_data['momentum'] * 100
        
        return {
            'risk_score': risk_score,
            'return_potential': return_potential,
            'allocation': max(min(return_potential / max(risk_score, 1) * 10, 20), 5),
            'risks': ['Market volatility', 'Sector-specific risks'],
            'strategies': ['Diversification', 'Stop-loss orders']
        }

    def optimize_portfolio(self, stocks_analysis):
        """Simple portfolio optimization"""
        if len(stocks_analysis) < 2:
            return None
        
        # Calculate weights based on risk-adjusted returns
        weights = {}
        total_score = 0
        
        for symbol, data in stocks_analysis.items():
            ai_analysis = data.get('ai_analysis', {})
            risk_score = ai_analysis.get('risk_score', 5)
            return_potential = ai_analysis.get('return_potential', 10)
            
            # Risk-adjusted score (higher is better)
            score = return_potential / max(risk_score, 1)
            weights[symbol] = score
            total_score += score
        
        # Normalize weights
        if total_score > 0:
            weights = {k: (v / total_score) * 100 for k, v in weights.items()}
        else:
            # Equal weighting if no scores
            equal_weight = 100 / len(stocks_analysis)
            weights = {k: equal_weight for k in stocks_analysis.keys()}
        
        return weights

def create_price_chart(df, symbol):
    """Create interactive price chart with technical indicators"""
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
    
    # Moving averages
    if 'sma_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20',
                      line=dict(color='orange', dash='dash')),
            row=1, col=1
        )
    
    if 'sma_50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50',
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
    
    # Bollinger Bands
    if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper',
                      line=dict(color='gray', dash='dot'), showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower',
                      line=dict(color='gray', dash='dot'), showlegend=False,
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
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
    
    fig.update_layout(height=600, showlegend=True,
                     template="plotly_white")
    
    return fig

def create_technical_chart(df):
    """Create technical indicators chart"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('RSI', 'MACD')
    )
    
    # RSI
    if 'rsi' in df.columns:
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
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd'], name='MACD',
                      line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd_signal'], name='Signal',
                      line=dict(color='red')),
            row=2, col=1
        )
    
    fig.update_layout(height=400, showlegend=True,
                     template="plotly_white")
    
    return fig

def main():
    st.markdown('<h1 class="main-header">🤖 AI Stock Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = StockAnalyzer()
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # API Key input
    deepseek_key = st.sidebar.text_input("DeepSeek API Key (optional)", 
                                        type="password",
                                        help="Get your API key from https://platform.deepseek.com/")
    if deepseek_key:
        st.session_state.deepseek_api_key = deepseek_key
    
    # Stock selection
    st.sidebar.subheader("Stock Selection")
    default_symbols = "AAPL, MSFT, GOOGL, TSLA, NVDA, AMZN"
    symbols_input = st.sidebar.text_area("Stock Symbols (comma separated)", 
                                        value=default_symbols,
                                        help="Enter stock symbols separated by commas")
    
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    
    # Analysis period
    period = st.sidebar.selectbox("Analysis Period", 
                                 ["3mo", "6mo", "1y", "2y", "5y"], 
                                 index=2)
    
    # Analyze button
    if st.sidebar.button("🚀 Analyze Stocks", use_container_width=True):
        with st.spinner("Analyzing stocks... This may take a few moments."):
            st.session_state.analysis_results = {}
            
            progress_bar = st.progress(0)
            for i, symbol in enumerate(symbols):
                st.write(f"📊 Analyzing {symbol}...")
                
                # Get stock data
                stock_data = st.session_state.analyzer.get_stock_data(symbol, period)
                if stock_data is None:
                    continue
                
                # Calculate technical indicators
                technical_signals, df_with_indicators = st.session_state.analyzer.calculate_technical_indicators(
                    stock_data['historical']
                )
                
                # AI Analysis
                ai_analysis = st.session_state.analyzer.analyze_with_deepseek(
                    symbol, technical_signals, stock_data.get('info', {})
                )
                
                # Store results
                st.session_state.analysis_results[symbol] = {
                    'technical': technical_signals,
                    'ai_analysis': ai_analysis,
                    'historical': stock_data['historical'],
                    'df_with_indicators': df_with_indicators,
                    'info': stock_data.get('info', {})
                }
                
                progress_bar.progress((i + 1) / len(symbols))
    
    # Main content
    if st.session_state.analysis_results:
        display_results(st.session_state.analysis_results, symbols)
    else:
        display_welcome()

def display_welcome():
    """Display welcome message and instructions"""
    st.markdown("""
    ## Welcome to the AI Stock Analysis Dashboard! 🎯
    
    This tool helps you analyze stocks for **risk-adjusted returns** using:
    
    - **Real-time data** from Yahoo Finance 📊
    - **Technical analysis** with multiple indicators 🔍
    - **AI-powered insights** from DeepSeek 🤖
    - **Portfolio optimization** recommendations 💼
    
    ### How to use:
    1. Enter your DeepSeek API key in the sidebar (optional)
    2. Add stock symbols you want to analyze
    3. Click "Analyze Stocks"
    4. View detailed analysis and recommendations
    
    ### Example symbols to try:
    - Technology: AAPL, MSFT, GOOGL, NVDA
    - E-commerce: AMZN, SHOP
    - EVs: TSLA, NIO
    - Finance: JPM, V, MA
    """)

def display_results(analysis_results, symbols):
    """Display analysis results"""
    
    # Portfolio Optimization
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
                    value=f"{weight:.1f}%",
                    help=f"Recommended portfolio allocation for {symbol}"
                )
        
        with col2:
            # Portfolio pie chart
            fig = go.Figure(data=[go.Pie(
                labels=list(portfolio_weights.keys()),
                values=list(portfolio_weights.values()),
                hole=.3,
                textinfo='label+percent'
            )])
            fig.update_layout(title="Portfolio Allocation")
            st.plotly_chart(fig, use_container_width=True)
    
    # Individual Stock Analysis
    st.header("🔍 Individual Stock Analysis")
    
    for symbol in symbols:
        if symbol not in analysis_results:
            continue
            
        data = analysis_results[symbol]
        technical = data['technical']
        ai_analysis = data['ai_analysis']
        
        st.markdown(f"## {symbol}")
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            delta_color = "normal"
            if technical['rsi_signal'] == 'OVERSOLD':
                delta_color = "inverse"
            elif technical['rsi_signal'] == 'OVERBOUGHT':
                delta_color = "off"
                
            st.metric(
                "RSI",
                f"{technical['rsi']:.1f}",
                technical['rsi_signal'],
                delta_color=delta_color
            )
        
        with col2:
            st.metric(
                "MACD Signal",
                technical['macd_signal'],
                help="MACD trading signal"
            )
        
        with col3:
            st.metric(
                "Trend",
                technical['trend'],
                help="Short-term vs long-term trend"
            )
        
        with col4:
            st.metric(
                "Volatility",
                f"{(technical['volatility'] * 100):.1f}%",
                help="Annualized volatility"
            )
        
        with col5:
            momentum_pct = technical['momentum'] * 100
            st.metric(
                "Momentum",
                f"{momentum_pct:.1f}%",
                help="Weighted momentum score"
            )
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            price_chart = create_price_chart(data['df_with_indicators'], symbol)
            st.plotly_chart(price_chart, use_container_width=True)
        
        with col_chart2:
            tech_chart = create_technical_chart(data['df_with_indicators'])
            st.plotly_chart(tech_chart, use_container_width=True)
        
        # AI Analysis
        st.subheader("🤖 AI Risk-Return Analysis")
        
        if 'analysis' in ai_analysis:
            # Raw analysis text
            st.info(ai_analysis['analysis'])
        else:
            # Structured analysis
            col_ai1, col_ai2, col_ai3 = st.columns(3)
            
            with col_ai1:
                risk_score = ai_analysis.get('risk_score', 5)
                st.metric(
                    "Risk Score",
                    f"{risk_score}/10",
                    help="Lower is better"
                )
                st.progress(risk_score / 10)
            
            with col_ai2:
                return_potential = ai_analysis.get('return_potential', 10)
                st.metric(
                    "Return Potential",
                    f"{return_potential:.1f}%",
                    help="Expected return potential"
                )
            
            with col_ai3:
                allocation = ai_analysis.get('allocation', 10)
                st.metric(
                    "Recommended Allocation",
                    f"{allocation:.1f}%",
                    help="Suggested portfolio allocation"
                )
            
            # Risks and strategies
            col_risk, col_strat = st.columns(2)
            
            with col_risk:
                st.write("**Key Risks:**")
                risks = ai_analysis.get('risks', ['Not specified'])
                for risk in risks:
                    st.write(f"• {risk}")
            
            with col_strat:
                st.write("**Mitigation Strategies:**")
                strategies = ai_analysis.get('strategies', ['Not specified'])
                for strategy in strategies:
                    st.write(f"• {strategy}")
        
        st.markdown("---")

if __name__ == "__main__":
    main()
