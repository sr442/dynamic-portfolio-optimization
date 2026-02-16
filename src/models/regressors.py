from .base import BasePredictor
import pandas as pd
import numpy as np
# from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV # Removed to save size
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

# Using lightweight custom Mini-ML implementations to avoid scikit-learn dependency (~60MB+).
try:
    from .mini_ml import RandomForestRegressor, GradientBoostingRegressor
    HAS_MINI_ML = True
except ImportError:
    HAS_MINI_ML = False
    
HAS_SKLEARN = False # Force false to use Mini-ML

class XGBoostPredictor(BasePredictor):
    """
    ML Regressor: Predicts future returns using Gradient Boosting (sklearn) or Random Forest.
    Note: Named XGBoostPredictor for backward compatibility, but uses sklearn.ensemble.GradientBoostingRegressor.
    """
    def __init__(self, target_name: str = 'target_return', hyperparams: Optional[Dict[str, Any]] = None, model_type: str = 'auto'):
        super().__init__(name="XGBoostPredictor", params=hyperparams)
        self.target_name = target_name
        self.logger = logging.getLogger(__name__)
        
        if not HAS_MINI_ML:
            self.logger.error("Mini-ML module not available. Predictor disabled.")
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
        DISABLED to reduce dependencies.
        """
        self.logger.warning("Hyperparameter tuning is disabled in this lightweight version.")
        pass

