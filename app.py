import streamlit as st
import yfinance as yf
import requests
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import io

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Stock Portfolio Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Professional Dark Theme --------------------
def apply_professional_theme():
    st.markdown("""
        <style>
            /* Professional dark color palette */
            :root {
                --primary-bg: #0f1419;
                --secondary-bg: #1a1f29;
                --surface-bg: #242b3d;
                --accent-primary: #4f46e5;
                --accent-secondary: #06b6d4;
                --text-primary: #ffffff;
                --text-secondary: #94a3b8;
                --text-muted: #64748b;
                --border-color: #334155;
                --success-color: #10b981;
                --warning-color: #f59e0b;
                --error-color: #ef4444;
                --hover-bg: #2d3748;
            }
            
            /* Global app styling */
            .stApp {
                background: linear-gradient(135deg, var(--primary-bg) 0%, var(--secondary-bg) 100%);
                color: var(--text-primary);
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            /* Sidebar professional styling */
            .css-1d391kg {
                background: linear-gradient(180deg, var(--secondary-bg) 0%, var(--surface-bg) 100%);
                border-right: 2px solid var(--border-color);
            }
            
            /* Enhanced metric containers */
            div[data-testid="metric-container"] {
                background: linear-gradient(145deg, var(--surface-bg), var(--hover-bg));
                border: 1px solid var(--border-color);
                border-radius: 16px;
                padding: 1.5rem;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                backdrop-filter: blur(10px);
            }
            
            div[data-testid="metric-container"]:hover {
                transform: translateY(-4px);
                box-shadow: 0 12px 40px rgba(79, 70, 229, 0.3);
                border-color: var(--accent-primary);
            }
            
            /* Force high contrast text visibility */
            div[data-testid="metric-container"] label,
            div[data-testid="metric-container"] .metric-label {
                color: var(--text-secondary) !important;
                font-weight: 600 !important;
                font-size: 0.875rem !important;
                text-transform: uppercase !important;
                letter-spacing: 0.05em !important;
            }
            
            div[data-testid="metric-container"] div[data-testid="metric-value"],
            div[data-testid="metric-container"] .metric-value {
                color: var(--text-primary) !important;
                font-weight: 700 !important;
                font-size: 1.5rem !important;
            }
            
            /* Delta styling */
            div[data-testid="metric-container"] [data-testid="metric-delta"] {
                font-weight: 600 !important;
                font-size: 0.875rem !important;
            }
            
            /* Professional tab styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 4px;
                background: var(--surface-bg);
                border-radius: 12px;
                padding: 4px;
            }
            
            .stTabs [data-baseweb="tab"] {
                background: transparent;
                border-radius: 8px;
                border: none;
                color: var(--text-muted);
                font-weight: 600;
                font-size: 0.875rem;
                padding: 12px 24px;
                transition: all 0.2s ease;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
                color: white;
                box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
            }
            
            .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
                background: var(--hover-bg);
                color: var(--text-secondary);
            }
            
            /* Typography improvements */
            h1, h2, h3, h4, h5, h6 {
                color: var(--text-primary) !important;
                font-weight: 700 !important;
                letter-spacing: -0.025em;
            }
            
            h1 { font-size: 2.25rem !important; }
            h2 { font-size: 1.875rem !important; }
            h3 { font-size: 1.5rem !important; }
            h4 { font-size: 1.25rem !important; }
            
            /* Enhanced form controls */
            .stSelectbox > div > div,
            .stTextInput > div > div > input {
                background: var(--surface-bg) !important;
                border: 2px solid var(--border-color) !important;
                border-radius: 12px !important;
                color: var(--text-primary) !important;
                font-weight: 500;
                transition: all 0.2s ease;
            }
            
            .stSelectbox > div > div:focus-within,
            .stTextInput > div > div:focus-within {
                border-color: var(--accent-primary) !important;
                box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
            }
            
            /* File uploader styling */
            .stFileUploader > div {
                background: var(--surface-bg);
                border: 2px dashed var(--border-color);
                border-radius: 16px;
                transition: all 0.2s ease;
            }
            
            .stFileUploader > div:hover {
                border-color: var(--accent-primary);
                background: var(--hover-bg);
            }
            
            /* Professional alert styling */
            .stSuccess {
                background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05));
                border: 1px solid rgba(16, 185, 129, 0.3);
                border-radius: 12px;
                color: var(--success-color);
            }
            
            .stWarning {
                background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05));
                border: 1px solid rgba(245, 158, 11, 0.3);
                border-radius: 12px;
                color: var(--warning-color);
            }
            
            .stError {
                background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05));
                border: 1px solid rgba(239, 68, 68, 0.3);
                border-radius: 12px;
                color: var(--error-color);
            }
            
            .stInfo {
                background: linear-gradient(135deg, rgba(6, 182, 212, 0.1), rgba(6, 182, 212, 0.05));
                border: 1px solid rgba(6, 182, 212, 0.3);
                border-radius: 12px;
                color: var(--accent-secondary);
            }
            
            /* Enhanced dataframe styling */
            .stDataFrame {
                background: var(--surface-bg);
                border-radius: 12px;
                overflow: hidden;
                border: 1px solid var(--border-color);
            }
            
            /* Button enhancements */
            .stButton > button {
                background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
                color: white;
                border: none;
                border-radius: 12px;
                font-weight: 600;
                padding: 0.75rem 1.5rem;
                transition: all 0.2s ease;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-size: 0.875rem;
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 24px rgba(79, 70, 229, 0.4);
            }
            
            /* Scrollbar styling */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: var(--primary-bg);
            }
            
            ::-webkit-scrollbar-thumb {
                background: var(--border-color);
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: var(--text-muted);
            }
        </style>
    """, unsafe_allow_html=True)

apply_professional_theme()

# -------------------- Professional Header --------------------
st.markdown("""
    <div style="
        text-align: center; 
        padding: 3rem 0 2rem 0; 
        border-bottom: 2px solid var(--border-color); 
        margin-bottom: 2rem;
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.1), rgba(6, 182, 212, 0.1));
        border-radius: 16px;
        margin: -1rem -1rem 2rem -1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    ">
        <h1 style="
            background: linear-gradient(135deg, #4f46e5, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem; 
            margin: 0;
            font-weight: 800;
            letter-spacing: -0.05em;
        ">Stock Portfolio Dashboard</h1>
        <p style="
            color: var(--text-secondary); 
            font-size: 1.25rem; 
            margin-top: 1rem;
            font-weight: 500;
        ">Professional Investment Tracking & Portfolio Analytics</p>
    </div>
""", unsafe_allow_html=True)

# -------------------- Stocks by Sector --------------------
stocks = {
    "Nifty 50 Leaders": ["RELIANCE.NS", "INFY.NS", "TCS.NS"],
    "Banking Sector": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS"],
    "Pharmaceuticals": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS"],
    "Information Technology": ["WIPRO.NS", "TECHM.NS", "HCLTECH.NS"],
    "Automotive": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS"]
}

# -------------------- Sample Portfolio Data --------------------
def load_sample_portfolio():
    """Generate professional sample portfolio data"""
    portfolio_data = {
        'Stock': ['RELIANCE.NS', 'INFY.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
                 'SUNPHARMA.NS', 'WIPRO.NS', 'MARUTI.NS'],
        'Shares': [50, 100, 75, 80, 120, 60, 150, 40],
        'Purchase_Price': [2450.0, 1320.0, 3580.0, 1680.0, 920.0, 1150.0, 480.0, 9200.0],
        'Sector': ['Energy', 'IT', 'IT', 'Banking', 'Banking', 'Pharma', 'IT', 'Auto']
    }
    return pd.DataFrame(portfolio_data)

# -------------------- Professional News Sources --------------------
def fetch_professional_news():
    """Fetch professional market news updates"""
    news_items = [
        {
            "title": "Indian Equity Markets Demonstrate Resilience Amid Global Market Volatility",
            "description": "Domestic equity indices continue to outperform international markets, supported by strong institutional flows and robust corporate earnings across key sectors.",
            "source": "Financial Analytics",
            "publishedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "url": "#"
        },
        {
            "title": "Banking Sector Consolidation Drives Institutional Investment Interest",
            "description": "Major banking institutions report strong quarterly performance as digital transformation initiatives and credit growth support sector expansion.",
            "source": "Banking Industry Report",
            "publishedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "url": "#"
        },
        {
            "title": "Technology Sector Leadership in Digital Infrastructure Development",
            "description": "Leading IT services companies capitalize on enterprise digital transformation demand, reporting significant contract wins and revenue growth.",
            "source": "Technology Business Review",
            "publishedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "url": "#"
        },
        {
            "title": "Pharmaceutical Industry Expansion Through R&D Investment and Export Growth",
            "description": "Pharmaceutical companies demonstrate strong performance driven by international market expansion and increased research development spending.",
            "source": "Healthcare Industry Analysis",
            "publishedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "url": "#"
        },
        {
            "title": "Automotive Sector Transformation with Electric Vehicle Integration",
            "description": "Traditional automotive manufacturers accelerate electric vehicle production capabilities while maintaining strong conventional vehicle sales.",
            "source": "Automotive Industry News",
            "publishedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "url": "#"
        },
        {
            "title": "Market Outlook: Institutional Analysts Maintain Positive Growth Projections",
            "description": "Investment research firms continue to recommend domestic equity exposure based on strong macroeconomic fundamentals and policy support.",
            "source": "Investment Research",
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

# -------------------- Enhanced Chart Plotting --------------------
def plot_professional_stock_chart(ticker):
    df = fetch_stock_price(ticker)
    if df is not None:
        fig = go.Figure()
        
        # Add professional candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=ticker,
            increasing_line_color='#10b981',
            decreasing_line_color='#ef4444',
            increasing_fillcolor='rgba(16, 185, 129, 0.3)',
            decreasing_fillcolor='rgba(239, 68, 68, 0.3)'
        ))
        
        fig.update_layout(
            title={
                'text': f'{ticker.replace(".NS", "")} - Professional Stock Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#ffffff'}
            },
            xaxis_title='Trading Period',
            yaxis_title='Price (INR)',
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(36, 43, 61, 0.5)',
            font=dict(color='#ffffff', family='Inter'),
            xaxis=dict(
                gridcolor='rgba(51, 65, 85, 0.5)',
                showgrid=True,
                linecolor='#334155'
            ),
            yaxis=dict(
                gridcolor='rgba(51, 65, 85, 0.5)',
                showgrid=True,
                linecolor='#334155'
            ),
            margin=dict(l=0, r=0, t=60, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"Market data unavailable for {ticker}")

# -------------------- Professional Portfolio Analysis --------------------
def create_professional_portfolio_analysis(portfolio_df):
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
    
    # Create professional sector distribution
    sector_values = portfolio_df.groupby('Sector')['Current_Value'].sum().reset_index()
    
    fig = px.pie(
        sector_values, 
        values='Current_Value', 
        names='Sector',
        title="Portfolio Sector Allocation Analysis",
        color_discrete_sequence=['#4f46e5', '#06b6d4', '#10b981', '#f59e0b', '#8b5cf6']
    )
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff', family='Inter'),
        title=dict(x=0.5, font=dict(size=18)),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig, portfolio_df

# -------------------- Professional Sidebar --------------------
with st.sidebar:
    st.markdown("### Dashboard Controls")
    
    # Portfolio file uploader
    st.markdown("#### Portfolio Data Management")
    uploaded_file = st.file_uploader(
        "Upload Portfolio CSV",
        type=['csv'],
        help="Required columns: Stock, Shares, Purchase_Price, Sector"
    )
    
    if uploaded_file:
        portfolio_df = pd.read_csv(uploaded_file)
        st.success("Portfolio data loaded successfully")
    else:
        st.info("Using sample portfolio data")
        portfolio_df = load_sample_portfolio()
    
    # Professional stats display
    st.markdown("#### Portfolio Overview")
    total_stocks = len(portfolio_df)
    total_sectors = portfolio_df['Sector'].nunique()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Holdings", total_stocks)
    with col2:
        st.metric("Sectors", total_sectors)
    
    # Additional controls
    st.markdown("#### Analysis Settings")
    show_advanced = st.checkbox("Advanced Analytics", value=False)
    refresh_data = st.button("Refresh Market Data")
    
    if refresh_data:
        st.cache_data.clear()
        st.success("Data refreshed")

# -------------------- Main Navigation Tabs --------------------
portfolio_tab, holdings_tab, analysis_tab, charts_tab, news_tab = st.tabs([
    "Live Portfolio", 
    "My Holdings",
    "Portfolio Analysis", 
    "Stock Charts", 
    "Market News"
])

# -------------------- Live Portfolio Tab --------------------
with portfolio_tab:
    st.markdown("## Live Market Overview")
    
    # Professional sector performance display
    st.markdown("### Sector Performance Dashboard")
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
                    st.warning(f"{ticker}: Data unavailable")

# -------------------- Holdings Tab --------------------
with holdings_tab:
    st.markdown("## Portfolio Holdings Management")
    
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
            st.metric("Total Investment", f"₹{total_investment:,.0f}")
        with col2:
            st.metric("Current Value", f"₹{total_current_value:,.0f}")
        with col3:
            st.metric("Total P&L", f"₹{total_pnl:,.0f}", f"{pnl_percentage:+.1f}%")
        with col4:
            st.metric("Total Holdings", f"{len(portfolio_df)} positions")
    
        # Professional holdings table
        st.markdown("### Detailed Holdings Analysis")
        
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
        
        # Professional performance analysis
        st.markdown("### Performance Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top Performers")
            top_gainers = enhanced_portfolio.nlargest(3, 'P&L_%')[['Stock', 'P&L_%']].copy()
            top_gainers['Stock'] = top_gainers['Stock'].str.replace('.NS', '')
            for _, row in top_gainers.iterrows():
                st.success(f"**{row['Stock']}**: +{row['P&L_%']:.1f}%")
        
        with col2:
            st.markdown("#### Underperformers")
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
    st.markdown("## Advanced Portfolio Analytics")
    
    if not portfolio_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sector distribution pie chart
            pie_fig, updated_portfolio = create_professional_portfolio_analysis(portfolio_df)
            st.plotly_chart(pie_fig, use_container_width=True)
        
        with col2:
            # Professional stock-wise analysis
            stock_fig = px.bar(
                updated_portfolio,
                x='Stock',
                y='Current_Value',
                color='Sector',
                title="Individual Stock Value Distribution",
                color_discrete_sequence=['#4f46e5', '#06b6d4', '#10b981', '#f59e0b', '#8b5cf6']
            )
            stock_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(36, 43, 61, 0.5)',
                font=dict(color='#ffffff', family='Inter'),
                xaxis=dict(tickangle=45, title='Stock Symbol'),
                yaxis=dict(title='Portfolio Value (₹)'),
                title=dict(x=0.5, font=dict(size=16))
            )
            st.plotly_chart(stock_fig, use_container_width=True)
        
        # Advanced performance metrics
        st.markdown("### Advanced Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_current = updated_portfolio['Current_Value'].sum()
            total_invested = (portfolio_df['Shares'] * portfolio_df['Purchase_Price']).sum()
            total_gain_loss = total_current - total_invested
            st.metric("Total Return", f"₹{total_gain_loss:,.2f}", f"{(total_gain_loss/total_invested)*100:+.2f}%")
        
        with col2:
            best_performer = updated_portfolio.loc[
                updated_portfolio['Current_Value'].idxmax(), 'Stock'
            ].replace('.NS', '')
            st.metric("Top Position", best_performer)
        
        with col3:
            largest_holding = updated_portfolio.loc[
                updated_portfolio['Current_Value'].idxmax(), 'Current_Value'
            ]
            st.metric("Largest Holding", f"₹{largest_holding:,.2f}")
        
        with col4:
            diversification = len(updated_portfolio['Sector'].unique())
            st.metric("Sector Diversification", f"{diversification} sectors")

# -------------------- Charts Tab --------------------
with charts_tab:
    st.markdown("## Professional Stock Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_sector = st.selectbox("Select Sector", list(stocks.keys()))
    with col2:
        selected_stock = st.selectbox("Select Stock", stocks[selected_sector])
    
    plot_professional_stock_chart(selected_stock)
    
    # Professional technical analysis
    df = fetch_stock_price(selected_stock)
    if df is not None:
        st.markdown("### Technical Analysis Indicators")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sma_5 = df['Close'].rolling(5).mean().iloc[-1]
            st.metric("5-Day Moving Average", f"₹{sma_5:.2f}")
        
        with col2:
            volatility = df['Close'].pct_change().std() * 100
            st.metric("Price Volatility", f"{volatility:.2f}%")
        
        with col3:
            volume_avg = df['Volume'].mean()
            st.metric("Average Volume", f"{volume_avg/1000000:.1f}M")

# -------------------- News Tab --------------------
with news_tab:
    st.markdown("## Market Intelligence & News")
    
    # Professional news fetching
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
                articles = fetch_professional_news()
                st.info("Displaying curated market intelligence")
        else:
            articles = fetch_professional_news()
            st.info("Displaying curated market intelligence")
            
    except Exception as e:
        articles = fetch_professional_news()
        st.info("Displaying curated market intelligence")
    
    # Professional news display
    if articles:
        for i in range(0, len(articles), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(articles):
                    article = articles[i + j]
                    with col:
                        # Professional source handling
                        source = article.get('source', 'Market Intelligence')
                        if isinstance(source, dict):
                            source_name = source.get('name', 'Market Intelligence')
                        else:
                            source_name = source
                        
                        st.markdown(f"""
                            <div style="
                                background: linear-gradient(145deg, var(--surface-bg), var(--hover-bg));
                                padding: 2rem;
                                border-radius: 16px;
                                border: 1px solid var(--border-color);
                                margin-bottom: 1.5rem;
                                height: 300px;
                                overflow: hidden;
                                position: relative;
                                transition: all 0.3s ease;
                                backdrop-filter: blur(10px);
                            ">
                                <h4 style="
                                    background: linear-gradient(135deg, #4f46e5, #06b6d4);
                                    -webkit-background-clip: text;
                                    -webkit-text-fill-color: transparent;
                                    margin-top: 0; 
                                    line-height: 1.4;
                                    font-weight: 700;
                                    font-size: 1.1rem;
                                ">
                                    {article['title'][:85]}{'...' if len(article['title']) > 85 else ''}
                                </h4>
                                <p style="
                                    color: var(--text-secondary); 
                                    font-size: 0.95rem; 
                                    line-height: 1.5; 
                                    margin-bottom: 3.5rem;
                                    font-weight: 400;
                                ">
                                    {article['description'][:130] if article.get('description') else 'Professional market analysis and investment insights for portfolio management'}...
                                </p>
                                <div style="
                                    position: absolute; 
                                    bottom: 1.5rem; 
                                    left: 2rem; 
                                    right: 2rem;
                                ">
                                    <a href="{article.get('url', '#')}" target="_blank" 
                                       style="
                                           background: linear-gradient(135deg, #4f46e5, #06b6d4);
                                           -webkit-background-clip: text;
                                           -webkit-text-fill-color: transparent;
                                           text-decoration: none; 
                                           font-weight: 600;
                                           font-size: 0.9rem;
                                       ">
                                       Read Full Analysis →
                                    </a>
                                    <p style="
                                        color: var(--text-muted); 
                                        font-size: 0.8rem; 
                                        margin-top: 0.75rem;
                                        font-weight: 500;
                                    ">
                                        {source_name} • 
                                        {article.get('publishedAt', datetime.now().strftime('%Y-%m-%d %H:%M'))}
                                    </p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
    else:
        st.warning("Market news temporarily unavailable. Please try again later.")

# -------------------- Professional Footer --------------------
st.markdown("---")
st.markdown("""
    <div style="
        text-align: center; 
        padding: 2.5rem 0; 
        color: var(--text-muted);
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.05), rgba(6, 182, 212, 0.05));
        border-radius: 16px;
        margin-top: 2rem;
    ">
        <h4 style="
            background: linear-gradient(135deg, #4f46e5, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
            font-weight: 700;
        ">Professional Stock Portfolio Dashboard</h4>
        <p style="font-size: 1rem; margin-bottom: 0.5rem; font-weight: 500;">
            Powered by yFinance • Plotly • Streamlit • Professional Analytics
        </p>
        <p style="font-size: 0.875rem; color: var(--text-muted);">
            Upload your portfolio CSV or analyze sample data for comprehensive investment insights
        </p>
    </div>
""", unsafe_allow_html=True)