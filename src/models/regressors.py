from .base import BasePredictor
import pandas as pd
import numpy as np
try:
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
except ImportError:
    pass
import logging
from typing import Optional, Dict, Any

# Try importing xgboost, fallback to sklearn if missing (common on fresh Mac environments without libomp)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception as e:
    HAS_XGBOOST = False

try:
    from sklearn.ensemble import RandomForestRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class XGBoostPredictor(BasePredictor):
    """
    ML Regressor: Predicts future returns using XGBoost (or RandomForest fallback).
    """
    def __init__(self, target_name: str = 'target_return', hyperparams: Optional[Dict[str, Any]] = None, model_type: str = 'auto'):
        super().__init__(name="XGBoostPredictor", params=hyperparams)
        self.target_name = target_name
        self.logger = logging.getLogger(__name__)
        
        # Determine model type
        if model_type == 'random_forest':
            use_xgboost = False
        elif model_type == 'xgboost':
            use_xgboost = True
        else: # auto
            use_xgboost = HAS_XGBOOST

        if use_xgboost and HAS_XGBOOST:
            self.hyperparams = hyperparams or {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'objective': 'reg:squarederror'
            }
            self.model = xgb.XGBRegressor(**self.hyperparams)
            self.model_type = 'xgboost'
        elif HAS_SKLEARN:
            if use_xgboost and not HAS_XGBOOST:
                self.logger.warning("XGBoost requested but not available. Falling back to RandomForestRegressor.")
            
            self.hyperparams = hyperparams or {
                'n_estimators': 100,
                'max_depth': 5
            }
            self.model = RandomForestRegressor(**self.hyperparams)
            self.model_type = 'random_forest'
        else:
            self.logger.error("Neither XGBoost nor Scikit-Learn available. XGBoostPredictor is disabled.")
            self.model = None
            self.model_type = 'disabled'

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the model.
        """
        self.logger.info(f"Training {self.model_type}...")
        self.model.fit(X, y)
        self.logger.info("Training complete.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict return.
        """
        preds = self.model.predict(X)
        return pd.Series(preds, index=X.index)

    def tune(self, X: pd.DataFrame, y: pd.Series, n_iter: int = 10):
        """
        Perform hyperparameter tuning.
        """
        if self.model_type == 'xgboost':
            param_dist = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
            estimator = xgb.XGBRegressor(objective='reg:squarederror')
        else:
            # Random Forest tuning space
            param_dist = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10]
            }
            estimator = RandomForestRegressor()

        tscv = TimeSeriesSplit(n_splits=5)
        
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='neg_mean_squared_error',
            cv=tscv,
            verbose=1,
            n_jobs=-1
        )
        
        search.fit(X, y)
        self.hyperparams.update(search.best_params_)
        self.model = search.best_estimator_
        self.logger.info(f"Tuning complete. Best params: {search.best_params_}")

