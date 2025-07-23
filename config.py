from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('ALPHA_VANTAGE_API_KEY', '')

# Popular ETF for comparison
popular_etf = 'SPY'

# Model parameters
model_params = {
    'fmm_n_components': 2,
    'xgboost_n_estimators': 100,
    'xgboost_random_state': 42
}

# Global dataset for caching
global_data = {'returns': None, 'volume': None, 'tickers': None}

sp100_tickers = ['AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AIG', 'ALL', 'AMGN', 'AMT', 'AMZN', 'AVGO', 'AXP', 'BA', 'BAC', 'BK', 'BKNG', 'BLK', 'BMY', 'BRK.B', 'C', 'CAT', 'CHTR', 'CL', 'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS', 'CVX', 'DHR', 'DIS', 'DOW', 'DUK', 'EMN', 'EMR', 'ETN', 'F', 'FCX', 'FDX', 'GD', 'GE', 'GILD', 'GM', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KHC', 'KO', 'LIN', 'LLY', 'LMT', 'LOW', 'MA', 'MCD', 'MDT', 'MET', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PM', 'PYPL', 'QCOM', 'RTX', 'SBUX', 'SCHW', 'SO', 'SPG', 'T', 'TGT', 'TMUS', 'TSLA', 'TXN', 'UNH', 'UNP', 'UPS', 'USB', 'V', 'VZ', 'WFC', 'WMT', 'XOM']