import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
import json

# Page config
st.set_page_config(
    page_title="AI Stock Dashboard: Maximize Risk-Adjusted Returns",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .stock-card { background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #1f77b4; }
    .positive { color: #00cc96; }
    .negative { color: #ef553b; }
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
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if "Time Series (Daily)" not in data:
            return None
        df = pd.DataFrame(data["Time Series (Daily)"]).T
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        }, inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Alpha Vantage price error: {e}")
        return None

@st.cache_data(ttl=7200)
def fetch_fundamentals(symbol):
    try:
        api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
        url = "https://www.alphavantage.co/query"
        params = {"function": "OVERVIEW", "symbol": symbol, "apikey": api_key}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if "Symbol" not in 
            return {}
        return {
            "pe": float(data.get("PERatio", 0) or 0),
            "eps": float(data.get("EPS", 0) or 0),
            "market_cap": data.get("MarketCapitalization", "N/A"),
            "sector": data.get("Sector", "N/A"),
            "name": data.get("Name", symbol)
        }
    except:
        return {}

@st.cache_data(ttl=7200)
def fetch_news_sentiment(symbol):
    try:
        api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "limit": 2,
            "apikey": api_key
        }
        resp = requests.get(url, params=params, timeout=10)
        feed = resp.json().get("feed", [])
        if not feed:
            return {"summary": "No recent news.", "sentiment": "Neutral", "score": 0.0}
        avg = np.mean([float(a.get("overall_sentiment_score", 0)) for a in feed])
        label = "Bullish" if avg > 0.15 else "Bearish" if avg < -0.15 else "Neutral"
        headlines = " | ".join([a["title"] for a in feed])[:150] + "..."
        return {"summary": headlines, "sentiment": label, "score": avg}
    except:
        return {"summary": "News unavailable.", "sentiment": "Neutral", "score": 0.0}

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

# === DeepSeek AI Analysis ===
def deepseek_analysis(symbol, technical, fundamentals, news, risk_score, return_potential):
    try:
        api_key = st.secrets["DEEPSEEK_API_KEY"]
    except KeyError:
        return {
            "reasoning": "⚠️ DEEPSEEK_API_KEY missing in secrets. Add it to enable AI analysis.",
            "risk_adjusted_outlook": "Medium",
            "recommended_allocation_percent": 5.0
        }

    prompt = f"""
You are a Chief Investment Officer optimizing for **maximum risk-adjusted returns** (Sharpe ratio).
Analyze {symbol} using the following inputs:

- Price: ${technical['current_price']:.2f} ({technical['price_change']:+.2f}%)
- Technicals: RSI={technical['rsi']:.1f} ({technical['rsi_signal']}), Trend={technical['trend']}, MACD={technical['macd_signal']}
- Volatility: {technical['volatility']:.1%} → Risk Score: {risk_score:.1f}/10
- Fundamentals: P/E={fundamentals.get('pe', 'N/A')}, EPS=${fundamentals.get('eps', 'N/A')}, Sector={fundamentals.get('sector', 'N/A')}
- News Sentiment: {news['sentiment']} (score: {news['score']:.2f})
- Return Potential: {return_potential:.1f}%

Provide a concise 2-3 sentence synthesis and recommend an allocation % (0–15%) for a diversified portfolio.
Output ONLY valid JSON with keys: "reasoning", "risk_adjusted_outlook" ("High"/"Medium"/"Low"), "recommended_allocation_percent" (number).
"""

    for attempt in range(3):
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 300,
                    "response_format": {"type": "json_object"}
                },
                timeout=20
            )
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                result = json.loads(content)
                if all(k in result for k in ["reasoning", "risk_adjusted_outlook", "recommended_allocation_percent"]):
                    result["recommended_allocation_percent"] = float(result["recommended_allocation_percent"])
                    return result
            time.sleep(2)
        except Exception as e:
            time.sleep(2)
    return {
        "reasoning": "❌ DeepSeek AI failed after retries. Check API key or network.",
        "risk_adjusted_outlook": "Medium",
        "recommended_allocation_percent": 5.0
    }

# === Charts ===
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

    # Sidebar
    st.sidebar.header("Configuration")
    symbols_input = st.sidebar.text_area("Stock Symbols (comma-separated)", "AAPL, MSFT, GOOGL", height=100)
    period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=2)
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

    if st.sidebar.button("🚀 Analyze with DeepSeek AI", type="primary"):
        if not symbols:
            st.warning("Enter at least one stock symbol.")
            return

        results = {}
        progress = st.progress(0)
        for i, sym in enumerate(symbols):
            st.write(f"🔍 Analyzing {sym}...")
            df = fetch_price_data(sym)
            if df is None or len(df) < 20:
                st.error(f"No price data for {sym}")
                continue

            # Slice to period
            days_map = {"1mo":21, "3mo":63, "6mo":126, "1y":252}
            df = df.tail(days_map[period])
            tech_signals, df_ind = calculate_technical_indicators(df)
            fundamentals = fetch_fundamentals(sym)
            news = fetch_news_sentiment(sym)

            # Risk & return scoring
            risk = min(tech_signals['volatility'] * 15, 10)
            trend_score = 1 if tech_signals['trend'] == 'BULLISH' else -1 if tech_signals['trend'] == 'BEARISH' else 0
            rsi_score = 0.8 if tech_signals['rsi_signal'] == 'OVERSOLD' else -0.2 if tech_signals['rsi_signal'] == 'OVERBOUGHT' else 0.5
            macd_score = 0.3 if tech_signals['macd_signal'] == 'BULLISH' else -0.3
            raw_return = (trend_score + rsi_score + macd_score) * 20
            return_potential = max(min(raw_return, 30), -10)

            # DeepSeek AI
            ai_result = deepseek_analysis(sym, tech_signals, fundamentals, news, risk, return_potential)

            results[sym] = {
                'tech': tech_signals,
                'fundamentals': fundamentals,
                'news': news,
                'df': df_ind,
                'risk': risk,
                'return_potential': return_potential,
                'ai': ai_result
            }

            progress.progress((i+1)/len(symbols))
            if (i+1) % 4 == 0 and i+1 < len(symbols):
                time.sleep(15)  # Respect Alpha Vantage rate limits

        st.session_state.results = results
        st.success("✅ Analysis complete with DeepSeek AI!")

    # Display Results
    if st.session_state.results:
        st.header("📊 DeepSeek AI: Risk-Adjusted Return Analysis")
        for sym, data in st.session_state.results.items():
            with st.container():
                st.subheader(f"{sym} — {data['ai'].get('risk_adjusted_outlook', 'N/A')} Outlook")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Price", f"${data['tech']['current_price']:.2f}", f"{data['tech']['price_change']:+.2f}%")
                col2.metric("RSI", f"{data['tech']['rsi']:.1f}", data['tech']['rsi_signal'])
                col3.metric("Volatility", f"{data['tech']['volatility']:.1%}")
                col4.metric("Allocation", f"{data['ai'].get('recommended_allocation_percent', 0):.1f}%")

                st.plotly_chart(create_price_chart(data['df'], sym), use_container_width=True)
                st.markdown(f"**🧠 DeepSeek AI Reasoning:** {data['ai'].get('reasoning', 'N/A')}")
                st.markdown("---")
    else:
        st.info("Enter stock symbols and click **Analyze** to begin.")

if __name__ == "__main__":
    main()
