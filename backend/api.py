from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import pandas as pd
import numpy as np
import datetime
import logging
from .src.data.ingestion import DataIngestion
from .src.data.features import FeatureEngineer, MarketState
from .src.models.baseline import MomentumPredictor
from .src.models.risk import VolatilityPredictor
from .src.optimization.optimizer import MeanVarianceOptimizer, RiskParityOptimizer
from .src.backtest.engine import BacktestEngine

app = FastAPI(title="Dynamic Portfolio Optimization API", version="1.0.0")

# Allow CORS for frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class OptRequest(BaseModel):
    tickers: List[str]
    capital: float
    optimizer: str = "Mean-Variance"
    risk_aversion: float = 2.5
    lookback_days: int = 365

class BacktestRequest(BaseModel):
    tickers: List[str]
    capital: float
    start_date: str
    end_date: str
    optimizer: str = "Mean-Variance"
    risk_aversion: float = 2.5

class PortfolioWeight(BaseModel):
    ticker: str
    weight: float
    value: float

class OptResponse(BaseModel):
    allocation: List[PortfolioWeight]
    market_date: str
    message: str

# --- Dependency Cache ---
# Simple in-memory cache for data
DATA_CACHE = {}

def get_market_data(tickers: List[str], start: str, end: str):
    key = (tuple(sorted(tickers)), start, end)
    if key in DATA_CACHE:
        return DATA_CACHE[key]
    
    ingestor = DataIngestion(tickers=tickers, start_date=start, end_date=end)
    data = ingestor.fetch_data()
    if not data.empty:
        DATA_CACHE[key] = data
    return data

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.datetime.now().isoformat()}

@app.post("/api/optimize", response_model=OptResponse)
def optimize_portfolio(req: OptRequest):
    try:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=req.lookback_days)
        
        logger.info(f"Optimize request for {len(req.tickers)} tickers.")
        
        raw_data = get_market_data(req.tickers, str(start_date), str(end_date))
        
        if raw_data.empty:
            raise HTTPException(status_code=404, detail="No market data found for tickers.")
            
        fe = FeatureEngineer()
        features = fe.compute_features(raw_data)
        
        if features.empty:
             raise HTTPException(status_code=500, detail="Feature computation failed.")
             
        last_date = features.index[-1]
        ms = MarketState(features)
        state_now = ms.get_state(last_date)
        
        X_now = state_now.unstack(level='Feature')
        
        # Models
        predictor = MomentumPredictor(momentum_window=20)
        
        pred_returns = predictor.predict(X_now)
        
        # Covariance
        hist_features = features.iloc[-60:]
        hist_returns = hist_features.xs('log_ret', level='Feature', axis=1)
        cov_matrix = hist_returns.cov()
        
        # Optimizer
        if req.optimizer == "Mean-Variance":
            opt = MeanVarianceOptimizer(risk_aversion=req.risk_aversion)
        else:
            opt = RiskParityOptimizer()
            
        weights = opt.optimize(pred_returns, cov_matrix)
        
        # Format Response
        allocation = []
        for ticker, weight in weights.items():
            if weight > 0.001: # Filter tiny weights
                allocation.append(PortfolioWeight(
                    ticker=ticker,
                    weight=float(weight),
                    value=float(weight * req.capital)
                ))
        
        # Sort by weight desc
        allocation.sort(key=lambda x: x.weight, reverse=True)
        
        return OptResponse(
            allocation=allocation,
            market_date=str(last_date.date()),
            message="Optimization Successful"
        )
        
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backtest")
def run_backtest(req: BacktestRequest):
    try:
        logger.info(f"Backtest request: {req.start_date} to {req.end_date}")
        
        raw_data = get_market_data(req.tickers, req.start_date, req.end_date)
        
        if raw_data.empty:
            raise HTTPException(status_code=404, detail="No market data found.")
            
        fe = FeatureEngineer()
        predictor = MomentumPredictor(momentum_window=20)
        risk_model = VolatilityPredictor(window=60)
        
        if req.optimizer == "Mean-Variance":
            opt = MeanVarianceOptimizer(risk_aversion=req.risk_aversion)
        else:
            opt = RiskParityOptimizer()
            
        engine = BacktestEngine(
            initial_capital=req.capital,
            feature_engineer=fe,
            predictor_model=predictor,
            risk_model=risk_model,
            optimizer=opt,
            verbose=False
        )
        
        results = engine.run(raw_data, start_date=req.start_date, end_date=req.end_date)
        
        # Serialize results for JSON
        res_df = results['results'].reset_index()
        # Convert date to string
        res_df['Date'] = res_df['Date'].astype(str)
        
        metrics = results['metrics']
        # Handle nan in metrics
        metrics = {k: (v if not np.isnan(v) else 0) for k, v in metrics.items()}
        
        chart_data = res_df[['Date', 'PortfolioValue']].to_dict(orient='records')
        
        # Weights (sampled monthly to reduce payload?) or all points
        # For simplicity, send last 100 points
        weights_data = []
        for _, row in res_df.iloc[::5].iterrows(): # Downsample 5x
             w = row['Weights'] # Dict
             weights_data.append({"date": row['Date'], **w})

        return {
            "metrics": metrics,
            "chart_data": chart_data,
            "weights_data": weights_data
        }

    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
