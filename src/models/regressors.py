from .base import BasePredictor
import pandas as pd
import numpy as np
try:
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
except ImportError:
    pass
import logging
from typing import Optional, Dict, Any

from .base import BasePredictor
import pandas as pd
import numpy as np
try:
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
except ImportError:
    pass
import logging
from typing import Optional, Dict, Any

# Using Scikit-Learn GradientBoostingRegressor as a lightweight alternative to XGBoost
# to reduce deployment slug size (XGBoost binary is ~100MB).
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class XGBoostPredictor(BasePredictor):
    """
    ML Regressor: Predicts future returns using Gradient Boosting (sklearn) or Random Forest.
    Note: Named XGBoostPredictor for backward compatibility, but uses sklearn.ensemble.GradientBoostingRegressor.
    """
    def __init__(self, target_name: str = 'target_return', hyperparams: Optional[Dict[str, Any]] = None, model_type: str = 'auto'):
        super().__init__(name="XGBoostPredictor", params=hyperparams)
        self.target_name = target_name
        self.logger = logging.getLogger(__name__)
        
        if not HAS_SKLEARN:
            self.logger.error("Scikit-Learn not available. Predictor disabled.")
            self.model = None
            self.model_type = 'disabled'
            return

        # Determine model type
        # 'xgboost' request now maps to GradientBoostingRegressor
        if model_type == 'random_forest':
            self.model_type = 'random_forest'
            self.hyperparams = hyperparams or {
                'n_estimators': 100,
                'max_depth': 5,
                'random_state': 42
            }
            self.model = RandomForestRegressor(**self.hyperparams)
            
        else: # 'xgboost' or 'auto' defaults to Gradient Boosting
            self.model_type = 'xgboost' # Keep ID as xgboost for UI consistency
            self.hyperparams = hyperparams or {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'random_state': 42
            }
            self.model = GradientBoostingRegressor(**self.hyperparams)

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
        if self.model is None:
             return pd.Series(0, index=X.index)
             
        preds = self.model.predict(X)
        return pd.Series(preds, index=X.index)

    def tune(self, X: pd.DataFrame, y: pd.Series, n_iter: int = 10):
        """
        Perform hyperparameter tuning.
        """
        if self.model is None:
            return

        if isinstance(self.model, GradientBoostingRegressor):
            param_dist = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'max_features': ['sqrt', 'log2', None]
            }
            estimator = GradientBoostingRegressor(random_state=42)
        else:
            # Random Forest tuning space
            param_dist = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10]
            }
            estimator = RandomForestRegressor(random_state=42)

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

