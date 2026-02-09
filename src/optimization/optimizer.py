from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import cvxpy as cp
import logging
from typing import Optional, Dict, Any, List, Union
from .utils import make_psd, clean_weights

class BaseOptimizer(ABC):
    """
    Abstract base class for Portfolio Optimizers.
    """
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def optimize(self, 
                 expected_returns: pd.Series, 
                 covariance_matrix: pd.DataFrame, 
                 current_weights: Optional[pd.Series] = None) -> pd.Series:
        """
        Calculate optimal weights.
        """
        pass

class RiskParityOptimizer(BaseOptimizer):
    """
    Risk Parity Optimizer (Convex Formulation).
    Minimizes: 0.5 * x^T * Sigma * x - sum(log(x))
    Weights = x / sum(x)
    """
    def __init__(self):
        super().__init__(name="RiskParity")

    def optimize(self, 
                 expected_returns: pd.Series, 
                 covariance_matrix: pd.DataFrame, 
                 current_weights: Optional[pd.Series] = None) -> pd.Series:
        
        Sigma = make_psd(covariance_matrix.values) # Ensure PSD
        n = len(Sigma)
        
        x = cp.Variable(n)
        
        # Risk Parity Formulation
        # 0.5 * x.T @ Sigma @ x - sum(log(x))
        # Wait, cp.quad_form(x, Sigma) returns scaler
        risk = cp.quad_form(x, Sigma)
        log_barrier = cp.sum(cp.log(x))
        
        obj = cp.Minimize(0.5 * risk - log_barrier)
        constraints = [x >= 0]
        
        try:
             prob = cp.Problem(obj, constraints)
             prob.solve()
        except cp.SolverError:
             self.logger.error("Risk Parity Solver failed. Falling back to 1/N.")
             return pd.Series(1/n, index=expected_returns.index)
        
        if x.value is None or np.any(np.isnan(x.value)):
            self.logger.warning("Risk Parity Infeasible. Falling back to 1/N.")
            return pd.Series(1/n, index=expected_returns.index)

        raw_weights = x.value
        weights = pd.Series(raw_weights / np.sum(raw_weights), index=expected_returns.index)
        return clean_weights(weights)

class MeanVarianceOptimizer(BaseOptimizer):
    """
    Classic Mean-Variance Optimization:
    Maximize: mu^T * w - lambda * w^T * Sigma * w
    Subject to: sum(w) = 1, w >= 0 (long only)
    """
    def __init__(self, risk_aversion: float = 1.0, long_only: bool = True):
        super().__init__(name="MeanVariance")
        self.risk_aversion = risk_aversion
        self.long_only = long_only

    def optimize(self, 
                 expected_returns: pd.Series, 
                 covariance_matrix: pd.DataFrame, 
                 current_weights: Optional[pd.Series] = None) -> pd.Series:
        
        mu = expected_returns.values
        # Ensure PSD for stability
        Sigma = make_psd(covariance_matrix.values)
        n = len(mu)
        
        w = cp.Variable(n)
        
        try:
             risk = cp.quad_form(w, Sigma)
             objective = cp.Maximize(mu @ w - self.risk_aversion * risk)
             
             constraints = [cp.sum(w) == 1]
             if self.long_only:
                 constraints.append(w >= 0)
             
             prob = cp.Problem(objective, constraints)
             prob.solve()
             
             if w.value is None:
                 raise ValueError("Optimization returned None")
                 
             weights = pd.Series(w.value, index=expected_returns.index)
             return clean_weights(weights)
             
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}. Falling back to Equal Weight.")
            return pd.Series(1/n, index=expected_returns.index)


class EqualWeightOptimizer(BaseOptimizer):
    """
    Baseline: 1/N allocation.
    """
    def __init__(self):
        super().__init__(name="EqualWeight")

    def optimize(self, expected_returns: pd.Series, covariance_matrix: pd.DataFrame, **kwargs) -> pd.Series:
        n = len(expected_returns)
        return pd.Series(1/n, index=expected_returns.index)
