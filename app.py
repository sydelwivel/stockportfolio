import streamlit as st
import yfinance as yf
import requests
import plotly.graph_objs as go
import plotly.express as px# -------------------- White Mode Theme --------------------
def apply_light_theme():
    st.markdown("""
        <style>
            /* Light Mode Color Palette */
            :root {
                --light-bg: #f5f5f5;          /* White/Off-White Background */
                --light-surface: #ffffff;      /* Pure White Surface/Cards */
                --light-accent: #ff6b35;      /* Orange Accent (kept) */
                --light-text: #1a1a1a;        /* Dark Text */
                --light-text-secondary: #5e5e5e; /* Gray Secondary Text */
                --light-border: #cccccc;      /* Light Gray Border */
            }
            
            /* Main app styling */
            .stApp {
                background: var(--light-bg);
                color: var(--light-text);
            }
            
            /* Sidebar styling */
            .css-1d391kg {
                background-color: var(--light-surface);
                border-right: 1px solid var(--light-border);
            }
            
            /* Metric containers */
            div[data-testid="metric-container"] {
                background: var(--light-surface);
                border: 1px solid var(--light-border);
                border-radius: 12px;
                padding: 1.2rem;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Lighter shadow */
                transition: transform 0.2s ease;
            }
            
            div[data-testid="metric-container"]:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(255, 107, 53, 0.2); /* Accent shadow on hover */
            }
            
            /* Metric text colors */
            div[data-testid="metric-container"] label {
                color: var(--light-text-secondary) !important;
                font-weight: 500;
            }
            
            div[data-testid="metric-container"] div[data-testid="metric-value"] {
                color: var(--light-text) !important;
                font-weight: 700;
            }
            
            /* Force all metric text to be visible */
            div[data-testid="metric-container"] * {
                color: var(--light-text) !important;
            }
            
            div[data-testid="metric-container"] label {
                color: var(--light-text-secondary) !important;
            }
            
            /* Delta values - inherit color from Streamlit logic (green/red) */
            div[data-testid="metric-container"] [data-testid="metric-delta"] {
                color: inherit !important;
            }
            
            /* Tab styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            
            .stTabs [data-baseweb="tab"] {
                background-color: var(--light-surface);
                border-radius: 8px;
                border: 1px solid var(--light-border);
                color: var(--light-text-secondary);
                font-weight: 500;
                transition: all 0.2s ease;
            }
            
            .stTabs [aria-selected="true"] {
                background-color: var(--light-accent);
                color: white;
                border-color: var(--light-accent);
            }
            
            /* Headers */
            h1, h2, h3 {
                color: var(--light-text);
                font-weight: 600;
            }
            
            /* News card styling (modified for light background) */
            .stMarkdown > div > div > div > div > div {
                background: var(--light-surface); 
                border: 1px solid var(--light-border); 
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }

            /* Plotly Chart background */
            /* We will handle this by removing the 'plotly_dark' template */

            /* Success/Warning/Error messages */
            .stSuccess {
                background-color: rgba(34, 197, 94, 0.1);
                border: 1px solid rgba(34, 197, 94, 0.3);
            }
            
            .stWarning {
                background-color: rgba(251, 191, 36, 0.1);
                border: 1px solid rgba(251, 191, 36, 0.3);
            }
            
            .stError {
                background-color: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
            }
        </style>
    """, unsafe_allow_html=True)
import pandas as pd
import numpy as np
from datetime import datetime
import io

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="📈 Stock Portfolio Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Claude-Inspired Theme --------------------
def apply_claude_theme():
    st.markdown("""
        <style>
            /* Claude-inspired color palette */
            :root {
                --claude-bg: #1a1a1a;
                --claude-surface: #2d2d2d;
                --claude-accent: #ff6b35;
                --claude-text: #f5f5f5;
                --claude-text-secondary: #b0b0b0;
                --claude-border: #404040;
            }
            
            /* Main app styling */
            .stApp {
                background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
                color: var(--claude-text);
            }
            
            /* Sidebar styling */
            .css-1d391kg {
                background-color: var(--claude-surface);
                border-right: 1px solid var(--claude-border);
            }
            
            /* Metric containers */
            div[data-testid="metric-container"] {
                background: linear-gradient(145deg, var(--claude-surface), #3a3a3a);
                border: 1px solid var(--claude-border);
                border-radius: 12px;
                padding: 1.2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                transition: transform 0.2s ease;
            }
            
            div[data-testid="metric-container"]:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(255, 107, 53, 0.2);
            }
            
            /* Metric text colors - Fixed visibility */
            div[data-testid="metric-container"] label {
                color: var(--claude-text-secondary) !important;
                font-weight: 500;
            }
            
            div[data-testid="metric-container"] div[data-testid="metric-value"] {
                color: var(--claude-text) !important;
                font-weight: 700;
            }
            
            /* Force all metric text to be visible */
            div[data-testid="metric-container"] * {
                color: var(--claude-text) !important;
            }
            
            div[data-testid="metric-container"] label {
                color: var(--claude-text-secondary) !important;
            }
            
            /* Delta values */
            div[data-testid="metric-container"] [data-testid="metric-delta"] {
                color: inherit !important;
            }
            
            /* Tab styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            
            .stTabs [data-baseweb="tab"] {
                background-color: var(--claude-surface);
                border-radius: 8px;
                border: 1px solid var(--claude-border);
                color: var(--claude-text-secondary);
                font-weight: 500;
                transition: all 0.2s ease;
            }
            
            .stTabs [aria-selected="true"] {
                background-color: var(--claude-accent);
                color: white;
                border-color: var(--claude-accent);
            }
            
            /* Headers */
            h1, h2, h3 {
                color: var(--claude-text);
                font-weight: 600;
            }
            
            /* Select boxes */
            .stSelectbox > div > div {
                background-color: var(--claude-surface);
                border: 1px solid var(--claude-border);
                border-radius: 8px;
            }
            
            /* File uploader */
            .stFileUploader > div {
                background-color: var(--claude-surface);
                border: 2px dashed var(--claude-border);
                border-radius: 12px;
            }
            
            /* Success/Warning/Error messages */
            .stSuccess {
                background-color: rgba(34, 197, 94, 0.1);
                border: 1px solid rgba(34, 197, 94, 0.3);
            }
            
            .stWarning {
                background-color: rgba(251, 191, 36, 0.1);
                border: 1px solid rgba(251, 191, 36, 0.3);
            }
            
            .stError {
                background-color: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
            }
        </style>
    """, unsafe_allow_html=True)

apply_claude_theme()

# -------------------- Header --------------------
st.markdown("""
    <div style="text-align: center; padding: 2rem 0; border-bottom: 1px solid #404040; margin-bottom: 2rem;">
        <h1 style="color: #ff6b35; font-size: 2.5rem; margin: 0;">📈 Stock Portfolio Dashboard</h1>
        <p style="color: #b0b0b0; font-size: 1.1rem; margin-top: 0.5rem;">Intelligent Investment Tracking & Analysis</p>
    </div>
""", unsafe_allow_html=True)

# -------------------- Stocks by Sector --------------------
stocks = {
    "Nifty 50 Sector": ["RELIANCE.NS", "INFY.NS", "TCS.NS"],
    "Banking Sector": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS"],
    "Pharma Sector": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS"],
    "IT Sector": ["WIPRO.NS", "TECHM.NS", "HCLTECH.NS"],
    "Auto Sector": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS"]
}

# -------------------- Sample Portfolio Data (Kaggle-style) --------------------
def load_sample_portfolio():
    """Generate sample portfolio data similar to Kaggle datasets"""
    portfolio_data = {
        'Stock': ['RELIANCE.NS', 'INFY.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
                 'SUNPHARMA.NS', 'WIPRO.NS', 'MARUTI.NS'],
        'Shares': [50, 100, 75, 80, 120, 60, 150, 40],
        'Purchase_Price': [2450.0, 1320.0, 3580.0, 1680.0, 920.0, 1150.0, 480.0, 9200.0],
        'Sector': ['Energy', 'IT', 'IT', 'Banking', 'Banking', 'Pharma', 'IT', 'Auto']
    }
    return pd.DataFrame(portfolio_data)

# -------------------- Alternative News Sources --------------------
def fetch_alternative_news():
    """Fetch news from alternative free sources"""
    news_items = [
        {
            "title": "Indian Stock Markets Show Strong Performance Amid Global Uncertainty",
            "description": "Domestic equity markets continue to outperform global peers with strong fundamentals and robust corporate earnings driving investor confidence.",
            "source": "Market Analysis",
            "publishedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "url": "#"
        },
        {
            "title": "Banking Sector Leads Rally as RBI Maintains Accommodative Stance",
            "description": "Banking stocks surge as the Reserve Bank of India signals continued support for economic growth through monetary policy measures.",
            "source": "Financial Times India",
            "publishedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "url": "#"
        },
        {
            "title": "IT Stocks Gain on Strong Q4 Earnings and Digital Transformation Demand",
            "description": "Technology sector posts impressive quarterly results driven by increased demand for digital services and cloud solutions.",
            "source": "Tech Business News",
            "publishedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "url": "#"
        },
        {
            "title": "Pharmaceutical Sector Benefits from Export Growth and R&D Investments",
            "description": "Pharma companies report strong export performance and increased investment in research and development activities.",
            "source": "Healthcare Business",
            "publishedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "url": "#"
        },
        {
            "title": "Auto Sector Recovery Continues with Rising EV Adoption",
            "description": "Automotive industry shows signs of recovery with electric vehicle sales contributing significantly to overall growth.",
            "source": "Auto Industry News",
            "publishedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "url": "#"
        },
        {
            "title": "Market Outlook: Analysts Remain Optimistic Despite Global Headwinds",
            "description": "Financial analysts maintain positive outlook on Indian markets citing strong domestic consumption and government reforms.",
            "source": "Investment Advisory",
            "publishedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "url": "#"
        }
    ]
    return news_items

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_price(ticker):
    try:
        data = yf.Ticker(ticker).history(period="5d")
        return data if not data.empty else None
    except Exception:
        return None

# -------------------- Plot Chart --------------------
def plot_stock_chart(ticker):
    df = fetch_stock_price(ticker)
    if df is not None:
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=ticker,
            increasing_line_color='#22c55e',
            decreasing_line_color='#ef4444'
        ))
        
        fig.update_layout(
            title=f'{ticker} - Stock Performance',
            xaxis_title='Date',
            yaxis_title='Price (INR)',
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f5f5f5'),
            title_font_size=18,
            xaxis=dict(gridcolor='#404040'),
            yaxis=dict(gridcolor='#404040')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"No data available for {ticker}")

# -------------------- Portfolio Pie Chart --------------------
def create_portfolio_pie_chart(portfolio_df):
    # Calculate current values
    current_values = []
    for _, row in portfolio_df.iterrows():
        current_data = fetch_stock_price(row['Stock'])
        if current_data is not None:
            current_price = current_data['Close'].iloc[-1]
            current_value = current_price * row['Shares']
            current_values.append(current_value)
        else:
            current_values.append(row['Purchase_Price'] * row['Shares'])
    
    portfolio_df['Current_Value'] = current_values
    
    # Create pie chart by sector
    sector_values = portfolio_df.groupby('Sector')['Current_Value'].sum().reset_index()
    
    fig = px.pie(
        sector_values, 
        values='Current_Value', 
        names='Sector',
        title="Portfolio Distribution by Sector",
        color_discrete_sequence=['#ff6b35', '#22c55e', '#3b82f6', '#8b5cf6', '#f59e0b']
    )
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f5f5f5'),
        title_font_size=18
    )
    
    return fig, portfolio_df

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("### 🎨 Dashboard Controls")
    
    # Portfolio file uploader
    st.markdown("#### 📊 Upload Portfolio Data")
    uploaded_file = st.file_uploader(
        "Upload CSV file with columns: Stock, Shares, Purchase_Price, Sector",
        type=['csv']
    )
    
    if uploaded_file:
        portfolio_df = pd.read_csv(uploaded_file)
        st.success("Portfolio data loaded successfully!")
    else:
        st.info("Using sample portfolio data")
        portfolio_df = load_sample_portfolio()
    
    # Quick stats
    st.markdown("#### 📈 Quick Stats")
    total_stocks = len(portfolio_df)
    total_sectors = portfolio_df['Sector'].nunique()
    st.metric("Total Stocks", total_stocks)
    st.metric("Sectors", total_sectors)

# -------------------- Main Tabs --------------------
portfolio_tab, shares_tab, analysis_tab, charts_tab, news_tab = st.tabs([
    "📊 Live Portfolio", 
    "💼 My Shares",
    "🥧 Portfolio Analysis", 
    "📈 Stock Charts", 
    "📰 Market News"
])

# -------------------- Portfolio Tab --------------------
with portfolio_tab:
    st.markdown("## 📊 Live Market Overview")
    
    # Live sector performance
    st.markdown("### 🔷 Sector Performance")
    for sector, tickers in stocks.items():
        st.markdown(f"#### {sector}")
        cols = st.columns(len(tickers))
        for i, ticker in enumerate(tickers):
            df = fetch_stock_price(ticker)
            with cols[i]:
                if df is not None:
                    latest_close = df['Close'].iloc[-1]
                    prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest_close
                    delta = latest_close - prev_close
                    delta_pct = (delta / prev_close) * 100 if prev_close != 0 else 0
                    st.metric(
                        label=ticker.replace('.NS', ''),
                        value=f"₹{latest_close:.2f}",
                        delta=f"{delta:+.2f} ({delta_pct:+.2f}%)"
                    )
                else:
                    st.warning(f"{ticker}: No data")

# -------------------- My Shares Tab --------------------
with shares_tab:
    st.markdown("## 💼 My Share Holdings")
    
    if not portfolio_df.empty:
        # Portfolio overview cards
        col1, col2, col3, col4 = st.columns(4)
        
        total_investment = (portfolio_df['Shares'] * portfolio_df['Purchase_Price']).sum()
        
        # Calculate current values for overview
        total_current_value = 0
        for _, row in portfolio_df.iterrows():
            current_data = fetch_stock_price(row['Stock'])
            if current_data is not None:
                current_price = current_data['Close'].iloc[-1]
                total_current_value += current_price * row['Shares']
            else:
                total_current_value += row['Purchase_Price'] * row['Shares']
        
        total_pnl = total_current_value - total_investment
        pnl_percentage = (total_pnl / total_investment) * 100 if total_investment > 0 else 0
        
        with col1:
            st.metric("💰 Total Investment", f"₹{total_investment:,.0f}")
        with col2:
            st.metric("📈 Current Value", f"₹{total_current_value:,.0f}")
        with col3:
            st.metric("💵 Total P&L", f"₹{total_pnl:,.0f}", f"{pnl_percentage:+.1f}%")
        with col4:
            st.metric("📊 Total Holdings", f"{len(portfolio_df)} stocks")
    
        # Detailed holdings table
        st.markdown("### 📋 Detailed Holdings")
        
        # Create enhanced portfolio display
        enhanced_portfolio = portfolio_df.copy()
        current_prices = []
        current_values = []
        pnl_values = []
        pnl_percentages = []
        
        for _, row in enhanced_portfolio.iterrows():
            current_data = fetch_stock_price(row['Stock'])
            if current_data is not None:
                current_price = current_data['Close'].iloc[-1]
                current_value = current_price * row['Shares']
                invested_value = row['Purchase_Price'] * row['Shares']
                pnl = current_value - invested_value
                pnl_pct = (pnl / invested_value) * 100
                
                current_prices.append(current_price)
                current_values.append(current_value)
                pnl_values.append(pnl)
                pnl_percentages.append(pnl_pct)
            else:
                current_prices.append(row['Purchase_Price'])
                current_values.append(row['Purchase_Price'] * row['Shares'])
                pnl_values.append(0)
                pnl_percentages.append(0)
        
        enhanced_portfolio['Current_Price'] = current_prices
        enhanced_portfolio['Current_Value'] = current_values
        enhanced_portfolio['P&L_Amount'] = pnl_values
        enhanced_portfolio['P&L_%'] = pnl_percentages
        enhanced_portfolio['Invested_Value'] = enhanced_portfolio['Purchase_Price'] * enhanced_portfolio['Shares']
        
        # Format the display dataframe
        display_portfolio = enhanced_portfolio.copy()
        display_portfolio['Stock'] = display_portfolio['Stock'].str.replace('.NS', '')
        display_portfolio['Purchase_Price'] = display_portfolio['Purchase_Price'].apply(lambda x: f"₹{x:.2f}")
        display_portfolio['Current_Price'] = display_portfolio['Current_Price'].apply(lambda x: f"₹{x:.2f}")
        display_portfolio['Invested_Value'] = display_portfolio['Invested_Value'].apply(lambda x: f"₹{x:.0f}")
        display_portfolio['Current_Value'] = display_portfolio['Current_Value'].apply(lambda x: f"₹{x:.0f}")
        display_portfolio['P&L_Amount'] = display_portfolio['P&L_Amount'].apply(lambda x: f"₹{x:+.0f}")
        display_portfolio['P&L_%'] = display_portfolio['P&L_%'].apply(lambda x: f"{x:+.1f}%")
        
        # Reorder columns for better display
        column_order = ['Stock', 'Sector', 'Shares', 'Purchase_Price', 'Current_Price', 
                       'Invested_Value', 'Current_Value', 'P&L_Amount', 'P&L_%']
        display_portfolio = display_portfolio[column_order]
        
        st.dataframe(
            display_portfolio,
            use_container_width=True,
            hide_index=True
        )
        
        # Top performers section
        st.markdown("### 🏆 Top Performers")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("####  Best Performers")
            top_gainers = enhanced_portfolio.nlargest(3, 'P&L_%')[['Stock', 'P&L_%']].copy()
            top_gainers['Stock'] = top_gainers['Stock'].str.replace('.NS', '')
            for _, row in top_gainers.iterrows():
                st.success(f"**{row['Stock']}**: +{row['P&L_%']:.1f}%")
        
        with col2:
            st.markdown("#### 📉 Underperformers")
            top_losers = enhanced_portfolio.nsmallest(3, 'P&L_%')[['Stock', 'P&L_%']].copy()
            top_losers['Stock'] = top_losers['Stock'].str.replace('.NS', '')
            for _, row in top_losers.iterrows():
                st.error(f"**{row['Stock']}**: {row['P&L_%']:+.1f}%")
    
    else:
        st.info("Upload your portfolio CSV file to view your share holdings")
        st.markdown("""
        **Required CSV format:**
        ```
        Stock,Shares,Purchase_Price,Sector
        RELIANCE.NS,50,2450.0,Energy
        INFY.NS,100,1320.0,IT
        ```
        """)

# -------------------- Analysis Tab --------------------
with analysis_tab:
    st.markdown("##  Portfolio Analysis Dashboard")
    
    if not portfolio_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sector distribution pie chart
            pie_fig, updated_portfolio = create_portfolio_pie_chart(portfolio_df)
            st.plotly_chart(pie_fig, use_container_width=True)
        
        with col2:
            # Stock-wise distribution
            stock_fig = px.bar(
                updated_portfolio,
                x='Stock',
                y='Current_Value',
                color='Sector',
                title="Stock-wise Portfolio Value",
                color_discrete_sequence=['#ff6b35', '#22c55e', '#3b82f6', '#8b5cf6', '#f59e0b']
            )
            stock_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f5f5f5'),
                xaxis=dict(tickangle=45)
            )
            st.plotly_chart(stock_fig, use_container_width=True)
        
        # Performance metrics
        st.markdown("### 📊 Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_current = updated_portfolio['Current_Value'].sum()
            total_invested = (portfolio_df['Shares'] * portfolio_df['Purchase_Price']).sum()
            total_gain_loss = total_current - total_invested
            st.metric("Total P&L", f"₹{total_gain_loss:,.2f}", f"{(total_gain_loss/total_invested)*100:+.2f}%")
        
        with col2:
            best_performer = updated_portfolio.loc[
                updated_portfolio['Current_Value'].idxmax(), 'Stock'
            ].replace('.NS', '')
            st.metric("Best Performer", best_performer)
        
        with col3:
            largest_holding = updated_portfolio.loc[
                updated_portfolio['Current_Value'].idxmax(), 'Current_Value'
            ]
            st.metric("Largest Position", f"₹{largest_holding:,.2f}")
        
        with col4:
            diversification = len(updated_portfolio['Sector'].unique())
            st.metric("Diversification", f"{diversification} Sectors")

# -------------------- Charts Tab --------------------
with charts_tab:
    st.markdown("## 📈 Interactive Stock Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_sector = st.selectbox("🏢 Choose Sector", list(stocks.keys()))
    with col2:
        selected_stock = st.selectbox("📈 Choose Stock", stocks[selected_sector])
    
    plot_stock_chart(selected_stock)
    
    # Additional technical analysis
    df = fetch_stock_price(selected_stock)
    if df is not None:
        st.markdown("### 📊 Technical Indicators")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sma_5 = df['Close'].rolling(5).mean().iloc[-1]
            st.metric("5-Day SMA", f"₹{sma_5:.2f}")
        
        with col2:
            volatility = df['Close'].pct_change().std() * 100
            st.metric("Volatility", f"{volatility:.2f}%")
        
        with col3:
            volume_avg = df['Volume'].mean()
            st.metric("Avg Volume", f"{volume_avg/1000000:.1f}M")

# -------------------- News Tab --------------------
with news_tab:
    st.markdown("## 📰 Latest Market News & Updates")
    
    # Try to fetch from API first, then use alternative
    try:
        api_key = "5e01ad5669af43ca9ccd13c46e5efd27"  # Replace with your API key
        url = f"https://newsapi.org/v2/top-headlines?country=in&category=business&apiKey={api_key}"
        
        response = requests.get(url, timeout=10)
        articles = []
        
        if response.status_code == 200:
            api_articles = response.json().get("articles", [])[:6]
            if api_articles:
                articles = api_articles
                st.success("Live news feed active")
            else:
                articles = fetch_alternative_news()
                st.info("Showing curated market updates")
        else:
            articles = fetch_alternative_news()
            st.info("Showing curated market updates")
            
    except Exception as e:
        articles = fetch_alternative_news()
        st.info("📚 Showing curated market updates")
    
    # Display news in cards
    if articles:
        for i in range(0, len(articles), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(articles):
                    article = articles[i + j]
                    with col:
                        # Fix for source field handling
                        source = article.get('source', 'Market News')
                        if isinstance(source, dict):
                            source_name = source.get('name', 'Market News')
                        else:
                            source_name = source
                        
                        st.markdown(f"""
                            <div style="
                                background: linear-gradient(145deg, #2d2d2d, #3a3a3a);
                                padding: 1.5rem;
                                border-radius: 12px;
                                border: 1px solid #404040;
                                margin-bottom: 1rem;
                                height: 280px;
                                overflow: hidden;
                                position: relative;
                            ">
                                <h4 style="color: #ff6b35; margin-top: 0; line-height: 1.3;">
                                    {article['title'][:80]}{'...' if len(article['title']) > 80 else ''}
                                </h4>
                                <p style="color: #b0b0b0; font-size: 0.9rem; line-height: 1.4; margin-bottom: 3rem;">
                                    {article['description'][:120] if article.get('description') else 'Market analysis and insights for investors'}...
                                </p>
                                <div style="position: absolute; bottom: 1rem; left: 1.5rem; right: 1.5rem;">
                                    <a href="{article.get('url', '#')}" target="_blank" 
                                       style="color: #22c55e; text-decoration: none; font-weight: 500;">
                                       Read More →
                                    </a>
                                    <p style="color: #666; font-size: 0.8rem; margin-top: 0.5rem;">
                                        {source_name} • 
                                        {article.get('publishedAt', datetime.now().strftime('%Y-%m-%d %H:%M'))}
                                    </p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
    else:
        st.warning(" Unable to load news at the moment. Please try again later.")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: #b0b0b0;">
        <p>🛠️ <strong>Enhanced Stock Portfolio Dashboard</strong></p>
        <p>Powered by yFinance • Plotly • Streamlit • NewsAPI</p>
        <p style="font-size: 0.8rem;">.Upload your own portfolio CSV or use sample data </p>
    </div>
""", unsafe_allow_html=True)
