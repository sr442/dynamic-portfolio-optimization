import logging
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta

from src.data.ingestion import DataIngestion
from src.data.features import FeatureEngineer
from src.models.baseline import MomentumPredictor
from src.models.risk import VolatilityPredictor
from src.optimization.optimizer import MeanVarianceOptimizer, RiskParityOptimizer
from src.backtest.engine import BacktestEngine

def main():
    # Setup Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Starting Dynamic Portfolio Optimization Pipeline")

    # Configuration
    TICKERS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'SPY']
    START_DATE = '2020-01-01'
    END_DATE = '2023-01-01'
    
    # 1. Data Ingestion
    logger.info("Step 1: Ingesting Data...")
    ingestor = DataIngestion(tickers=TICKERS, start_date=START_DATE, end_date=END_DATE)
    raw_data = ingestor.fetch_data()
    
    if raw_data.empty:
        logger.error("No data fetched. Exiting.")
        return

    # 2. Initialize Components
    logger.info("Step 2: Initializing Models & Optimizer...")
    
    # Feature Engineering
    fe = FeatureEngineer()
    
    # Models
    # We use MomentumPredictor as baseline signal
    predictor = MomentumPredictor(momentum_window=20)
    risk_model = VolatilityPredictor(window=60)
    
    # Optimizer
    # Start with Mean Variance
    optimizer = MeanVarianceOptimizer(risk_aversion=2.5)
    
    # 3. run Backtest
    logger.info("Step 3: Running Backtest Loop...")
    engine = BacktestEngine(
        initial_capital=100_000,
        feature_engineer=fe,
        predictor_model=predictor,
        risk_model=risk_model,
        optimizer=optimizer
    )
    
    try:
        results = engine.run(raw_data, start_date=START_DATE, end_date=END_DATE)
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Report
    if 'metrics' in results:
        logger.info("Step 4: Performance Report")
        metrics = results['metrics']
        print("\n" + "="*40)
        print("PERFORMANCE METRICS")
        print("="*40)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("="*40 + "\n")
        
        # Save results
        results_df = results.get('results')
        if results_df is not None:
            results_df.to_csv('backtest_results.csv')
            logger.info("Results saved to backtest_results.csv")

if __name__ == "__main__":
    main()
