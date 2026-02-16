import sys
import os
sys.path.append(os.getcwd())

import logging
logging.basicConfig(level=logging.INFO)

try:
    from src.optimization.optimizer import EqualWeightOptimizer
    print("✅ EqualWeightOptimizer imported successfully.")
    opt = EqualWeightOptimizer()
    print("✅ EqualWeightOptimizer instantiated.")
except ImportError as e:
    print(f"❌ Failed to import EqualWeightOptimizer: {e}")

try:
    from src.models.regressors import XGBoostPredictor
    
    # Test XGBoost
    print("\n--- Testing XGBoost Selection ---")
    xgb_pred = XGBoostPredictor(model_type='xgboost')
    print(f"Instantiated XGBoostPredictor with model_type='xgboost'. Resulting model_type: {xgb_pred.model_type}")
    
    # Test Random Forest
    print("\n--- Testing Random Forest Selection ---")
    rf_pred = XGBoostPredictor(model_type='random_forest')
    print(f"Instantiated XGBoostPredictor with model_type='random_forest'. Resulting model_type: {rf_pred.model_type}")
    
    if rf_pred.model_type == 'random_forest':
        print("✅ Random Forest forced successfully.")
    else:
        print("❌ Failed to force Random Forest.")

except ImportError as e:
    print(f"❌ Failed to import XGBoostPredictor: {e}")
except Exception as e:
    print(f"❌ Error during model instantiation: {e}")
