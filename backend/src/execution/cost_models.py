from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Dict

class ExecutionModel(ABC):
    """
    Abstract Base Class for Execution models.
    """
    @abstractmethod
    def calculate_cost(self, current_weights: pd.Series, target_weights: pd.Series, portfolio_value: float) -> float:
        """
        Calculate total transaction cost.
        """
        pass

class SimpleExecutionModel(ExecutionModel):
    """
    Simple Execution Model: Fixed proportional cost (BP per dollar traded).
    """
    def __init__(self, transaction_cost_bps: float = 10.0):
        # 10 bps = 0.0010 (0.1%)
        self.transaction_cost_rate = transaction_cost_bps / 10000.0

    def calculate_cost(self, current_weights: pd.Series, target_weights: pd.Series, portfolio_value: float) -> float:
        """
        Cost = Turnover * Portfolio Value * Rate
        """
        # Ensure indices align (fill missing with 0)
        all_assets = current_weights.index.union(target_weights.index)
        w_curr = current_weights.reindex(all_assets, fill_value=0.0)
        w_target = target_weights.reindex(all_assets, fill_value=0.0)
        
        turnover = np.abs(w_target - w_curr).sum()  # This is total turnover (buy + sell).
        # Usually turnover definition: 0.5 * abs(diff), but cost applies to both buy and sell legs.
        # So sum(abs(buy)) + sum(abs(sell)) is correct.
        
        cost = turnover * portfolio_value * self.transaction_cost_rate
        return cost

    def simulate_trade(self, current_portfolio_value: float, current_weights: pd.Series, target_weights: pd.Series) -> Dict[str, float]:
        """
        Simulate the trade execution.
        Returns:
            - 'cost': Transaction cost in currency
            - 'new_portfolio_value': Value after cost
        """
        cost = self.calculate_cost(current_weights, target_weights, current_portfolio_value)
        return {
            'cost': cost,
            'new_portfolio_value': current_portfolio_value - cost
        }
