import pandas as pd
import numpy as np
from typing import Dict, Any

def calculate_performance_metrics(returns_series: pd.Series, period: str = 'daily') -> Dict[str, float]:
    """
    Calculate core performance metrics for a return series.
    """
    if returns_series.empty:
        return {}
        
    annualization_factor = 252 if period == 'daily' else 12 # Assume daily or monthly
    
    total_return = (1 + returns_series).prod() - 1
    cagr = (1 + total_return) ** (annualization_factor / len(returns_series)) - 1
    
    mean_return = returns_series.mean() * annualization_factor
    volatility = returns_series.std() * np.sqrt(annualization_factor)
    
    sharpe_ratio = mean_return / volatility if volatility > 0 else 0
    
    # Max Drawdown
    cumulative_returns = (1 + returns_series).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Sortino Ratio
    negative_returns = returns_series[returns_series < 0]
    downside_volatility = negative_returns.std() * np.sqrt(annualization_factor)
    sortino_ratio = mean_return / downside_volatility if downside_volatility > 0 else 0
    
    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown
    }

def calculate_turnover(weights_history: pd.DataFrame) -> float:
    """
    Calculate annualized turnover.
    Turnover = 0.5 * sum(abs(w_t - w_{t-1})) / T
    Usually defined as one-way turnover.
    """
    turnover_series = weights_history.diff().abs().sum(axis=1)
    # Annualize based on frequency (assume daily)
    avg_turnover = turnover_series.mean() * 252 # If daily turnover is X, annualized is X * 252? No.
    # Annualized turnover is sum of turnover / years.
    # Or average daily turnover * 252.
    
    return turnover_series.mean() / 2 * 252 # One-way turnover
