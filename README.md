# AI-Driven Dynamic Portfolio Optimization System

## Overview
This project implements an industry-grade, end-to-end AI system for dynamic portfolio optimization. It follows a modular 6-layer architecture designed for extensibility and realism.

### Architecture Layers
1.  **Data Ingestion**: Fetches market data (OHLCV) using `yfinance`.
2.  **State Construction**: Computes technical indicators and constructs the market state vector $S_t$.
3.  **Predictive Intelligence**: Generates return forecasts ($\hat{\mu}$) and risk estimates ($\hat{\Sigma}$) using ML models (XGBoost/RandomForest) and baseline approaches.
4.  **Optimization**: Solves for optimal portfolio weights using `cvxpy` (Mean-Variance, Risk Parity).
5.  **Execution**: Simulates transaction costs and realistic trade execution.
6.  **Backtesting & Governance**: Runs the strategy through a historical simulation loop and reports performance metrics.

## Installation

### Prerequisites
- Python 3.10+
- (Optional) `libomp` for XGBoost on Mac (`brew install libomp`)

### Setup
1.  Navigate to the project directory:
    ```bash
    cd dynamic_portfolio_optimization
    ```

2.  Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If XGBoost installation fails on Mac, the system will automatically fallback to RandomForestRegressor.*

## Usage

### Running the Backtest Pipeline
To run the full end-to-end backtest with default configuration:

```bash
python main.py
```

### Running the Interactive Dashboard
To see the system running in an interactive web interface:

```bash
streamlit run app.py
```
This will open a local web server (usually at `http://localhost:8501`).

### Running the Full-Stack Application (Industry Grade)
For a production-ready interface with separated backend and frontend:

1.  **Start the Backend (API)**:
    ```bash
    uvicorn backend.api:app --reload --port 8000
    ```

2.  **Start the Frontend (UI)**:
    Open a new terminal window:
    ```bash
    cd frontend
    npm run dev
    ```
    Open your browser at `http://localhost:5173`.
    
The full-stack app features a dark-mode professional dashboard, real-time optimization, and interactive equity curves.

### Deployment
- **GitHub**: See `DEPLOYMENT.md`.
- **Vercel**: See `VERCEL_DEPLOYMENT.md` for serverless deployment instructions.

### Configuration
You can modify `main.py` to change:
- `TICKERS`: List of assets.
- `START_DATE` / `END_DATE`.
- `optimizer`: Switch between `MeanVarianceOptimizer` and `RiskParityOptimizer`.
- `risk_aversion`: Adjust risk tolerance.

## Project Structure
```
dynamic_portfolio_optimization/
├── main.py                 # Entry point
├── requirements.txt        # Python dependencies
├── src/
│   ├── data/               # Layer 1: Ingestion & Features
│   ├── models/             # Layer 2: Prediction (Base, Regressors, Risk)
│   ├── optimization/       # Layer 3: Optimizers (MVO, RiskParity)
│   ├── execution/          # Layer 4: Cost Models
│   └── backtest/           # Layer 5: Engine & Metrics
└── tests/                  # (Optional) Unit tests
```

## Future Extensions
- **Deep Learning**: Add LSTM/Transformer models in `src/models/deep.py`.
- **Reinforcement Learning**: Implement RL agent in `src/models/rl.py`.
- **Live Trading**: Connect `src/execution` to a brokerage API (e.g., Alpaca).
