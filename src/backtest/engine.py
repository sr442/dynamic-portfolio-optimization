import pandas as pd
import numpy as np
import logging
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from typing import Optional, Dict, List

from src.data.features import FeatureEngineer, MarketState
from src.models.base import BasePredictor
from src.optimization.optimizer import BaseOptimizer
from src.execution.cost_models import ExecutionModel, SimpleExecutionModel
from src.backtest.metrics import calculate_performance_metrics, calculate_turnover

class BacktestEngine:
    """
    Event-driven-like Backtest Engine.
    Simulates the decision pipeline period by period.
    """
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 feature_engineer: Optional[FeatureEngineer] = None,
                 predictor_model: Optional[BasePredictor] = None,
                 risk_model: Optional[BasePredictor] = None,
                 optimizer: Optional[BaseOptimizer] = None,
                 execution_model: Optional[ExecutionModel] = None,
                 verbose: bool = True):
        
        self.initial_capital = initial_capital
        self.feature_engineer = feature_engineer
        self.predictor_model = predictor_model
        self.risk_model = risk_model
        self.optimizer = optimizer
        self.execution_model = execution_model or SimpleExecutionModel()
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

    def run(self, data: pd.DataFrame, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Run the backtest loop.
        data: Multi-Index OHLCV DataFrame (from DataIngestion).
        """
        self.logger.info("Starting Backtest...")
        
        # 1. Prepare Features
        # Compute all features upfront (vectorized) to save time in loop, unless model is online learning
        # For this version, assume batch learning or static model for simplicity, or we re-train periodically.
        
        if self.feature_engineer:
            self.logger.info("Computing features...")
            features = self.feature_engineer.compute_features(data)
            market_state = MarketState(features)
        else:
            # Assume data already has features? Not supported yet.
            raise ValueError("FeatureEngineer is required.")

        # Filter timeline
        timeline = features.index
        timeline = timeline[(timeline >= start_date) & (timeline <= end_date)]
        
        if len(timeline) == 0:
            self.logger.error("No simulation periods found.")
            return {}

        # Initialization
        portfolio_value = self.initial_capital
        current_weights = pd.Series(0.0, index=features.columns.levels[0]) # Start w/ cash? 
        # Actually, let's assume fully invested in first period or empty.
        
        history = []
        
        # Loop
        for i, t in enumerate(tqdm(timeline[:-1], disable=not self.verbose)):
            # t is the "decision time" (e.g. close of today). 
            # We predict for t+1. 
            # We rebalance at t (close) to hold for t+1.
            
            next_t = timeline[i+1]
            
            # 1. Get State at t
            try:
                state_t = market_state.get_state(t)
            except KeyError:
                continue

            # 2. Predict (Layer 2)
            # Create a DataFrame for prediction (need X for all assets at time t)
            # state_t is Series with MultiIndex (Ticker, Feature)
            # Unstack to get DataFrame: Index=Ticker, Columns=Feature
            X_t = state_t.unstack(level='Feature')
            
            # Predict Returns
            # Predict Returns
            if self.predictor_model:
                # --- ML Training (Walk-Forward) ---
                if hasattr(self.predictor_model, 'model_type') and self.predictor_model.model_type in ['xgboost', 'random_forest']:
                    # Train on moving window of last ~2 years (504 trading days)
                    window_size = 504
                    
                    # Prepare Training Data
                    features_slice = features.loc[:t].tail(window_size)
                        
                    # Helper to prepare X,y from features
                    try:
                        # Stack: Index=(Date, Ticker), Cols=Feature
                        stack = features_slice.stack(level='Ticker')
                        stack = stack.sort_index(level=0)
                        
                        # Create Target: shifted log_ret per ticker
                        returns_df = features_slice.xs('log_ret', level='Feature', axis=1)
                        future_returns = returns_df.shift(-1) # Row t has return t+1
                        
                        # Stack again
                        y_stack = future_returns.stack().reindex(stack.index)
                        
                        # Combine
                        valid_mask = ~y_stack.isna()
                        X_train = stack[valid_mask]
                        X_train = X_train.sort_index(axis=1)
                        y_train = y_stack[valid_mask]
                        
                        if len(X_train) > 30: # Min samples to train
                            self.predictor_model.fit(X_train, y_train)
                            
                            # Ensure prediction input is also sorted
                            X_pred = X_t.sort_index(axis=1)
                            pred_returns = self.predictor_model.predict(X_pred)
                        else:
                            self.logger.warning(f"Insufficient data for training ({len(X_train)} samples). Skipping prediction.")
                            pred_returns = pd.Series(0, index=X_t.index)
                            
                    except Exception as e:
                       self.logger.warning(f"Training failed at {t}: {e}")
                       pred_returns = pd.Series(0, index=X_t.index)
                else:
                    # Baseline / Already Trained Models (e.g. Momentum)
                    pred_returns = self.predictor_model.predict(X_t)
            else:
                pred_returns = pd.Series(0, index=X_t.index)
            
            # Predict Covariance
            # Construct historical returns for covariance estimation
            # We need lookback. Let's use 60 days prior to t.
            # Efficient way: slice main data 'log_ret'
            if self.risk_model:
                # If risk model supports predict_covariance
                 # For now, let's assume we use simple historical cov from data
                 pass

            # Extract historical returns for covariance
            # Using closing prices from data. xs helps.
            # But feature engineering already has 'log_ret'.
            # We can use market_state efficiently.
            
            # Get log returns up to t
            # This is slow inside loop if we copy large data.
            # Heuristic: slice last 60 days
            lookback_start = timeline[max(0, i-60)]
            hist_slice = features.loc[lookback_start:t]
            hist_returns = hist_slice.xs('log_ret', level='Feature', axis=1)
            
            # Current estimated covariance
            if hasattr(self.risk_model, 'predict_covariance'):
                 cov_matrix = self.risk_model.predict_covariance(hist_returns)
            else:
                 cov_matrix = hist_returns.cov()

            # 3. Optimize (Layer 3)
            if self.optimizer:
                target_weights = self.optimizer.optimize(pred_returns, cov_matrix, current_weights)
            else:
                target_weights = current_weights # No change

            # 4. Execution (Layer 4)
            # Calculate Transaction Costs
            cost = self.execution_model.calculate_cost(current_weights, target_weights, portfolio_value)
            
            # Update Portfolio Value (deduct cost)
            portfolio_value -= cost
            
            # 5. Realize Returns (t to t+1)
            # We hold target_weights from t to next_t
            # Get actual returns for next period
            # log_ret at next_t is ln(P_{t+1}/P_t)
            # So we can use features.loc[next_t, (Ticker, 'log_ret')]
            realized_log_rets = market_state.get_state(next_t).unstack()['log_ret']
            realized_simple_rets = np.exp(realized_log_rets) - 1
            
            # Portfolio Return = w^T * r
            port_ret = (target_weights * realized_simple_rets).sum()
            
            portfolio_value *= (1 + port_ret)
            
            # Store History
            history.append({
                'Date': next_t,
                'PortfolioValue': portfolio_value,
                'Return': port_ret,
                'Cost': cost,
                'Weights': target_weights.to_dict()
            })
            
            current_weights = target_weights

        # Compile Results
        results_df = pd.DataFrame(history).set_index('Date')
        metrics = calculate_performance_metrics(results_df['Return'])
        
        return {
            'results': results_df,
            'metrics': metrics
        }
