import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
from src.data.ingestion import DataIngestion
from src.data.features import FeatureEngineer
from src.models.baseline import MomentumPredictor
from src.models.risk import VolatilityPredictor
from src.optimization.optimizer import MeanVarianceOptimizer, RiskParityOptimizer
from src.backtest.engine import BacktestEngine

# Configure Logging (to console)
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Dynamic Portfolio Optimization", layout="wide")

st.title("🤖 AI-Driven Dynamic Portfolio Optimization")
st.markdown("""
This dashboard demonstrates an end-to-end AI portfolio management system.
It fetches data, generates signals, optimizes allocations, and simulates execution.
""")

# Configure Logging (to console)
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Dynamic Portfolio Optimization", layout="wide")

st.title("🤖 AI-Driven Dynamic Portfolio Optimization")
st.markdown("""
This dashboard demonstrates an end-to-end AI portfolio management system.
It fetches data, generates signals, optimizes allocations, and simulates execution.
""")


import datetime

# Configure Logging (to console)
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Dynamic Portfolio Optimization", layout="wide")

st.title("🤖 AI-Driven Portfolio Manager")
st.markdown("""
**System Status**: 🟢 Online | **Market Data**: Real-time (Yahoo Finance)
""")

# Tabs for different PM workflows
tab_live, tab_backtest = st.tabs(["🚀 Live Optimization & Rebalance", "📊 Strategy Backtest"])

# Common Configuration (Sidebar)
with st.sidebar.form("global_config"):
    st.header("Global Configuration")
    
    # Tickers
    default_tickers = "AAPL, MSFT, GOOG, AMZN, SPY, NVDA, TSLA, GLD, TLT"
    tickers_input = st.text_area("Investment Universe (Comma Separated)", default_tickers, help="Assets to consider for allocation.")
    
    # Optimizer Settings
    optimizer_type = st.selectbox("Optimizer Model", ["Mean-Variance", "Risk Parity"])
    risk_aversion = st.slider("Risk Aversion (Lambda)", 0.5, 10.0, 2.5, help="Higher = More conservative")
    
    st.markdown("---")
    submitted = st.form_submit_button("Update Configuration")

@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_live_data(tickers, lookback_days=365):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=lookback_days)
    ingestor = DataIngestion(tickers=tickers, start_date=str(start_date), end_date=str(end_date))
    return ingestor.fetch_data()

# --- LIVE OPTIMIZATION TAB ---
with tab_live:
    st.header("🌍 Current Market Optimization")
    st.caption("Generate optimal target weights based on the latest market state.")
    
    col_cap, col_act = st.columns([1, 1])
    with col_cap:
        portfolio_capital = st.number_input("Total Portfolio Value ($)", 10000, 10_000_000, 100_000, step=1000)
    
    if st.button("Generate Target Allocation", type="primary"):
        tickers_list = [t.strip() for t in tickers_input.split(',')]
        
        with st.spinner("Fetching latest market data..."):
            # Fetch last 1 year of data for robust stats
            raw_data = fetch_live_data(tickers_list, lookback_days=365)
            
            if raw_data.empty:
                st.error("Failed to fetch data.")
            else:
                # 1. Feature Engineering on latest data
                fe = FeatureEngineer()
                features = fe.compute_features(raw_data)
                
                # 2. Get state for the LAST available timestep (Today/Yesterday)
                last_date = features.index[-1]
                ms = MarketState(features)
                state_now = ms.get_state(last_date)
                
                st.subheader(f"Market State as of {last_date.date()}")
                
                # 3. Predict & Optimize
                # Prepare X for prediction
                X_now = state_now.unstack(level='Feature') # (N_assets, N_features)
                
                # Models
                predictor = MomentumPredictor(momentum_window=20)
                risk_model = VolatilityPredictor(window=60)
                
                # Predict Returns (Next Period)
                pred_returns = predictor.predict(X_now)
                
                # Predict Covariance (Historical lookback)
                # Slice last 60 days of returns
                hist_features = features.iloc[-60:]
                # Getting 'log_ret' properly handling MultiIndex columns
                # features columns: (Ticker, Feature)
                hist_returns = hist_features.xs('log_ret', level='Feature', axis=1)
                cov_matrix = hist_returns.cov() # Simple cov
                
                # Optimize
                if optimizer_type == "Mean-Variance":
                    opt = MeanVarianceOptimizer(risk_aversion=risk_aversion)
                else:
                    opt = RiskParityOptimizer()
                
                target_weights = opt.optimize(pred_returns, cov_matrix)
                
                # --- DISPLAY RESULTS ---
                
                # 1. Optimal Weights Display
                weights_df = target_weights.to_frame(name="Target Weight")
                weights_df["Target Value ($)"] = weights_df["Target Weight"] * portfolio_capital
                weights_df["Target Weight"] = weights_df["Target Weight"].apply(lambda x: f"{x:.2%}")
                weights_df["Target Value ($)"] = weights_df["Target Value ($)"].apply(lambda x: f"${x:,.2f}")
                
                c1, c2 = st.columns([2, 1])
                
                with c1:
                    st.subheader("Recommended Allocation")
                    # Pie Chart
                    fig_pie = px.pie(values=target_weights.values, names=target_weights.index, title="Optimal Portfolio Composition")
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with c2:
                    st.subheader("Trade Sheet")
                    st.dataframe(weights_df, use_container_width=True)
                    
                    # Stylized download button for Trade Sheet
                    csv = weights_df.to_csv().encode('utf-8')
                    st.download_button(
                        label="📥 Download Trade Instructions (CSV)",
                        data=csv,
                        file_name='portfolio_rebalance.csv',
                        mime='text/csv',
                    )

# --- BACKTEST TAB ---
with tab_backtest:
    st.header("🕰️ Strategy Verification (Backtest)")
    st.caption("Simulate how this strategy would have performed in the past.")

    # Dates specific to backtest
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        bt_start_date = st.date_input("Backtest Start", pd.to_datetime("2020-01-01"))
    with col_d2:
        bt_end_date = st.date_input("Backtest End", pd.to_datetime("2023-01-01"))
    
    if st.button("Run Historical Simulation"):
        # Logic similar to before, but reused
        tickers_list = [t.strip() for t in tickers_input.split(',')]
        
        with st.spinner("Running Simulation..."):
            ingestor = DataIngestion(tickers=tickers_list, start_date=str(bt_start_date), end_date=str(bt_end_date))
            raw_data = ingestor.fetch_data()
            
            # Models
            fe = FeatureEngineer()
            predictor = MomentumPredictor(momentum_window=20)
            risk_model = VolatilityPredictor(window=60)

            if optimizer_type == "Mean-Variance":
                optimizer = MeanVarianceOptimizer(risk_aversion=risk_aversion)
            else:
                optimizer = RiskParityOptimizer()

            engine = BacktestEngine(
                initial_capital=100000, # Fixed for backtest demo
                feature_engineer=fe,
                predictor_model=predictor,
                risk_model=risk_model,
                optimizer=optimizer,
                verbose=True
            )

            results = engine.run(raw_data, start_date=str(bt_start_date), end_date=str(bt_end_date))
            
            if results:
                res_df = results['results']
                metrics = results['metrics']

                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Return", f"{metrics.get('Total Return', 0):.2%}")
                m2.metric("CAGR", f"{metrics.get('CAGR', 0):.2%}")
                m3.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0):.2f}")
                m4.metric("Max Drawdown", f"{metrics.get('Max Drawdown', 0):.2%}")

                # Charts
                st.subheader("Equity Curve")
                df_plot = res_df.reset_index()
                fig_equity = px.line(df_plot, x='Date', y='PortfolioValue', title="Portfolio Value Generated")
                st.plotly_chart(fig_equity, use_container_width=True)

                st.subheader("Historical Allocation")
                weights_list = res_df['Weights'].tolist()
                weights_df = pd.DataFrame(weights_list, index=res_df.index).reset_index()
                weights_melted = weights_df.melt(id_vars='Date', var_name='Asset', value_name='Weight')
                fig_weights = px.area(weights_melted, x='Date', y='Weight', color='Asset', title="Positions Over Time")
                st.plotly_chart(fig_weights, use_container_width=True)

