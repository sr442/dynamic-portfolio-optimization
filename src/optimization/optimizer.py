from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
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
    Simpler Risk Parity (Inverse Volatility) to avoid heavy solvers.
    Weights are inversely proportional to asset volatility.
    """
    def __init__(self):
        super().__init__(name="RiskParity")

    def optimize(self, 
                 expected_returns: pd.Series, 
                 covariance_matrix: pd.DataFrame, 
                 current_weights: Optional[pd.Series] = None) -> pd.Series:
        
        if covariance_matrix.empty:
             return self._fallback(expected_returns)

        # Calculate volatilities (safe sqrt)
        variances = np.diag(covariance_matrix.values)
        if np.any(variances <= 0) or np.any(np.isnan(variances)):
             self.logger.warning("Invalid variances detected. Fallback to EW.")
             return self._fallback(expected_returns)
             
        volatilities = np.sqrt(variances)
        
        # Inverse Volatility
        # Handle zero volatility by replacing with mean or small epsilon, or just fallback
        if np.any(volatilities < 1e-8):
             volatilities = np.maximum(volatilities, 1e-8)
             
        inv_vol = 1.0 / volatilities
        
        # Normalize to sum to 1
        weights = inv_vol / np.sum(inv_vol)
        
        return pd.Series(weights, index=expected_returns.index)

    def _fallback(self, expected_returns):
         return pd.Series(1.0 / len(expected_returns), index=expected_returns.index)

class MeanVarianceOptimizer(BaseOptimizer):
    """
    Analytical Mean-Variance Optimization.
    Using standard closed-form solution for Tangency Portfolio approx.
    w = Sigma^-1 * mu
    Normalized to sum to 1.
    Negative weights clipped for Long-Only constraint.
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
        Sigma = covariance_matrix.values
        n = len(mu)
        
        # Add regularization to Sigma for stability
        Sigma += np.eye(n) * 1e-6
        
        try:
            # Check for NaNs/Infs in inputs
            if np.any(np.isnan(Sigma)) or np.any(np.isinf(Sigma)) or np.any(np.isnan(mu)):
                 raise ValueError("NaNs in inputs")

            # Check condition number to avoid singular matrix errors
            cond_num = np.linalg.cond(Sigma)
            if cond_num > 1e12: # Ill-conditioned
                 raise np.linalg.LinAlgError("Matrix ill-conditioned")

            # Unconstrained solution: w* propto Eq. (risk_aversion optional scaling)
            # Maximize mu^T w - 0.5 lambda w^T Sigma w
            # First order condition: mu - lambda Sigma w = 0  => w = (1/lambda) Sigma^-1 mu
            
            # Solve Sigma * w = mu
            w_unc = np.linalg.solve(Sigma, mu)
            
            # Apply Constraints Heuristically
            w = w_unc / self.risk_aversion # Scale by risk aversion
            
            if self.long_only:
                w = np.maximum(w, 0) # Clip negative
            
            # Normalize to sum to 1
            w_sum = np.sum(w)
            if w_sum > 1e-8:
                w = w / w_sum
            else:
                # Fallback to Equal Weight if all zero or invalid
                return self._fallback(expected_returns)
                
            return pd.Series(w, index=expected_returns.index)
              
        except (np.linalg.LinAlgError, ValueError) as e:
            self.logger.warning(f"Optimization failed ({e}). Falling back to Equal Weight.")
            return self._fallback(expected_returns)

    def _fallback(self, expected_returns):
         return pd.Series(1.0 / len(expected_returns), index=expected_returns.index)


class EqualWeightOptimizer(BaseOptimizer):
    """
    Baseline: 1/N allocation.
    """
    def __init__(self):
        super().__init__(name="EqualWeight")

    def optimize(self, expected_returns: pd.Series, covariance_matrix: pd.DataFrame, current_weights: Optional[pd.Series] = None, **kwargs) -> pd.Series:
        n = len(expected_returns)
        return pd.Series(1/n, index=expected_returns.index)
