import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time

st.set_page_config(
    page_title="AI Stock Dashboard: Maximize Risk-Adjusted Returns",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# === Alpha Vantage Data Fetching ===
@st.cache_data(ttl=3600)
def fetch_price_data(symbol):
    try:
        api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
    except KeyError:
        st.error("❌ Missing ALPHA_VANTAGE_API_KEY in secrets.")
        return None

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": api_key
    }

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()

            if "Error Message" in data:
                st.warning(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
                return None

            if "Time Series (Daily)" not in data:
                st.warning(f"No price data for {symbol} (attempt {attempt+1})")
                time.sleep(2)
                continue

            df = pd.DataFrame(data["Time Series (Daily)"]).T
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df.rename(columns={
                "1. open": "Open", "2. high": "High", "3. low": "Low",
                "4. close": "Close", "5. volume": "Volume"
            }, inplace=True)
            df.sort_index(inplace=True)
            return df

        except Exception as e:
            st.warning(f"Attempt {attempt+1} failed for {symbol}: {str(e)[:80]}")
            time.sleep(2)

    st.error(f"❌ Failed to fetch data for {symbol} after 3 attempts.")
    return None

@st.cache_data(ttl=7200)
def fetch_fundamentals(symbol):
    try:
        api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
        url = "https://www.alphavantage.co/query"
        params = {"function": "OVERVIEW", "symbol": symbol, "apikey": api_key}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        # Check if API returned valid data
        if "Symbol" not in data:
            st.warning(f"Fundamental data unavailable for {symbol}")
            return {}

        def safe_float(val, default="N/A"):
            """Converts value to float safely, returns default if conversion fails."""
            try:
                if val is None or val == "" or val == "None":
                    return default
                return float(val)
            except (ValueError, TypeError):
                return default

        # Build fundamentals dict safely
        fundamentals = {
            "name": data.get("Name", symbol),
            "sector": data.get("Sector", "N/A"),
            "pe": safe_float(data.get("PERatio")),
            "eps": safe_float(data.get("EPS")),
            "market_cap": data.get("MarketCapitalization", "N/A"),
            "dividendyield": safe_float(data.get("DividendYield"))
        }

        # Log if key data is missing
        if fundamentals["pe"] == "N/A" or fundamentals["eps"] == "N/A":
            st.info(f"Partial fundamental data for {symbol} — using available fields only.")

        return fundamentals

    except Exception as e:
        st.warning(f"Error fetching fundamentals for {symbol}: {e}")
        return {}

@st.cache_data(ttl=7200)
def fetch_news_sentiment(symbol):
    try:
        api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
        url = "https://www.alphavantage.co/query"
        params = {"function": "NEWS_SENTIMENT", "tickers": symbol, "limit": 2, "apikey": api_key}
        resp = requests.get(url, params=params, timeout=10)
        feed = resp.json().get("feed", [])
        if not feed:
            return {"sentiment": "Neutral", "score": 0.0}
        avg = np.mean([float(a.get("overall_sentiment_score", 0)) for a in feed])
        label = "Bullish" if avg > 0.15 else "Bearish" if avg < -0.15 else "Neutral"
        return {"sentiment": label, "score": avg}
    except:
        return {"sentiment": "Neutral", "score": 0.0}

# === Technical Indicators ===
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_technical_indicators(df):
    if len(df) < 30:
        return {}, df
    df = df.copy()
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
    df['rsi'] = calculate_rsi(df['Close'])
    macd, sig = calculate_macd(df['Close'])
    df['macd'] = macd
    df['macd_signal'] = sig
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    latest = df.iloc[-1]
    signals = {
        'rsi': latest['rsi'] if not pd.isna(latest['rsi']) else 50,
        'rsi_signal': 'OVERSOLD' if latest['rsi'] < 30 else 'OVERBOUGHT' if latest['rsi'] > 70 else 'NEUTRAL',
        'macd_signal': 'BULLISH' if latest['macd'] > latest['macd_signal'] else 'BEARISH',
        'trend': 'BULLISH' if latest['sma_20'] > latest['sma_50'] else 'BEARISH',
        'volatility': latest['volatility'] if not pd.isna(latest['volatility']) else 0.3,
        'current_price': latest['Close'],
        'price_change': ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100) if len(df) > 1 else 0
    }
    return signals, df

# === Rule-Based Multi-Agent Analysis (Inspired by TradingAgents) ===
def rule_based_analysis(symbol, technical, fundamentals, news, risk_score, return_potential):
    # Technical Analyst
    tech_score = 0
    if technical['trend'] == 'BULLISH': tech_score += 4
    elif technical['trend'] == 'BEARISH': tech_score -= 4
    if technical['rsi_signal'] == 'OVERSOLD': tech_score += 3
    elif technical['rsi_signal'] == 'OVERBOUGHT': tech_score -= 2
    if technical['macd_signal'] == 'BULLISH': tech_score += 3
    elif technical['macd_signal'] == 'BEARISH': tech_score -= 3

    # Fundamental Analyst
    fund_score = 0
    pe = fundamentals.get('pe', 0)
    if pe > 0:
        if pe < 15: fund_score += 3
        elif pe < 25: fund_score += 1
        elif pe > 40: fund_score -= 2
    else:
        st.info(f"No PE ratio available for {symbol} — skipping fundamental score.")
    
    # Sentiment Analyst
    sent_score = 2 if news['sentiment'] == 'Bullish' else -2 if news['sentiment'] == 'Bearish' else 0

    total_score = tech_score + fund_score + sent_score
    risk_adjusted_score = total_score - max(0, risk_score - 5)

    if risk_adjusted_score >= 6:
        outlook = "High"
        allocation = min(15.0, max(8.0, risk_adjusted_score * 1.2))
    elif risk_adjusted_score >= 2:
        outlook = "Medium"
        allocation = min(10.0, max(4.0, risk_adjusted_score * 1.0))
    else:
        outlook = "Low"
        allocation = max(0.0, min(3.0, risk_adjusted_score + 3))

    reasons = []
    if tech_score != 0: reasons.append(f"Technical {'strength' if tech_score > 0 else 'weakness'} ({tech_score:+.0f})")
    if fund_score != 0: reasons.append(f"Fundamental {'support' if fund_score > 0 else 'concern'} (P/E={pe:.1f})")
    if sent_score != 0: reasons.append(f"Sentiment is {news['sentiment'].lower()}")
    reasoning = "; ".join(reasons) if reasons else "Balanced signals."
    if risk_score > 7: reasoning += f" ⚠️ High volatility (risk={risk_score:.1f}/10)."

    return {
        "reasoning": reasoning,
        "risk_adjusted_outlook": outlook,
        "recommended_allocation_percent": round(allocation, 1)
    }

# === Chart ===
def create_price_chart(df, symbol):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f'{symbol} Price', 'Volume'), row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='#1f77b4')), row=1, col=1)
    if 'sma_20' in df.columns and not df['sma_20'].isna().all():
        fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', line=dict(dash='dash', color='orange')), row=1, col=1)
    colors = ['green' if o <= c else 'red' for o, c in zip(df['Open'], df['Close'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, opacity=0.6), row=2, col=1)
    fig.update_layout(height=400, template="plotly_white", margin=dict(t=50, l=0, r=0, b=0))
    return fig

# === Main App ===
def main():
    st.markdown('<h1 class="main-header">🤖 AI Stock Dashboard: Maximize Risk-Adjusted Returns</h1>', unsafe_allow_html=True)

    if 'results' not in st.session_state:
        st.session_state.results = {}

    st.sidebar.header("Configuration")
    symbols_input = st.sidebar.text_area("Stock Symbols", "AAPL, MSFT, GOOGL, TSLA", height=100)
    period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=2)
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

    if st.sidebar.button("🚀 Analyze Stocks", type="primary"):
        if not symbols:
            st.warning("Enter at least one symbol.")
            return

        results = {}
        progress = st.progress(0)
        total = len(symbols)

        for i, sym in enumerate(symbols):
            st.write(f"🔍 Analyzing {sym}...")
            df = fetch_price_data(sym)
            if df is None or len(df) < 20:
                st.error(f"Skipping {sym} due to missing data.")
                progress.progress((i + 1) / total)
                continue

            days_map = {"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252}
            df = df.tail(days_map[period])
            tech_signals, df_ind = calculate_technical_indicators(df)
            fundamentals = fetch_fundamentals(sym)
            news = fetch_news_sentiment(sym)

            risk = min(tech_signals['volatility'] * 15, 10)
            trend_score = 1 if tech_signals['trend'] == 'BULLISH' else -1 if tech_signals['trend'] == 'BEARISH' else 0
            rsi_score = 0.8 if tech_signals['rsi_signal'] == 'OVERSOLD' else -0.2 if tech_signals['rsi_signal'] == 'OVERBOUGHT' else 0.5
            macd_score = 0.3 if tech_signals['macd_signal'] == 'BULLISH' else -0.3
            return_potential = max(min((trend_score + rsi_score + macd_score) * 20, 30), -10)

            ai_result = rule_based_analysis(sym, tech_signals, fundamentals, news, risk, return_potential)

            results[sym] = {
                'tech': tech_signals,
                'fundamentals': fundamentals,
                'news': news,
                'df': df_ind,
                'ai': ai_result
            }

            progress.progress((i + 1) / total)

            # ⏱ Respect Alpha Vantage rate limit: max 5 req/min
            if (i + 1) % 5 == 0 and (i + 1) < total:
                st.info("⏸ Pausing 15 seconds to respect Alpha Vantage rate limits...")
                time.sleep(15)

        st.session_state.results = results
        st.success("✅ Analysis complete with rule-based multi-agent system!")

    if st.session_state.results:
        st.header("📊 Multi-Agent Analysis: Risk-Adjusted Return Focus")
        for sym, data in st.session_state.results.items():
            st.subheader(f"{sym} — {data['ai']['risk_adjusted_outlook']} Outlook")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Price", f"${data['tech']['current_price']:.2f}", f"{data['tech']['price_change']:+.2f}%")
            col2.metric("RSI", f"{data['tech']['rsi']:.1f}", data['tech']['rsi_signal'])
            col3.metric("Volatility", f"{data['tech']['volatility']:.1%}")
            col4.metric("Allocation", f"{data['ai']['recommended_allocation_percent']:.1f}%")
    # Display Fundamentals Section
            col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns(5)
            col_f1.metric("Company", fundamentals.get("name", "N/A"))
            col_f2.metric("Sector", fundamentals.get("sector", "N/A"))
            col_f3.metric("P/E Ratio", fundamentals.get("pe", "N/A"))
            col_f4.metric("EPS", fundamentals.get("eps", "N/A"))
            col_f5.metric("Market Cap", fundamentals.get("market_cap", "N/A"))
    # Add Dividend Yield if available
            if fundamentals.get("dividendyield") != "N/A":
                st.metric("Dividend Yield", f"{fundamentals['dividendyield']:.2%}")
            else:
                st.metric("Dividend Yield", "N/A")
            
            st.plotly_chart(create_price_chart(data['df'], sym), use_container_width=True)
            st.markdown(f"**🧠 Multi-Agent Reasoning:** {data['ai']['reasoning']}")
            st.markdown("---")                 
    else:
        st.info("Enter stock symbols and click **Analyze** to begin.")

if __name__ == "__main__":
    main()
