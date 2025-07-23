import requests
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skewnorm, kstest, norm, t, chi2
from scipy.optimize import minimize
from datetime import datetime, timedelta
import xgboost as xgb
import time
from config import global_data, api_key, popular_etf

def fetch_alpha_vantage_data(ticker: str) -> pd.DataFrame:
    """
    Fetch daily stock data from Alpha Vantage API.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        pd.DataFrame: DataFrame with OHLC and volume data, or empty if failed.
    """
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}&outputsize=full'
    try:
        time.sleep(12)  # Respect 5 requests/minute limit
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'Time Series (Daily)' in data:
            df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df[['1. open', '2. high', '3. low', '4. close', '5. volume']].astype(float)
            return df
        else:
            return pd.DataFrame()
    except requests.exceptions.RequestException:
        return pd.DataFrame()

def reset_cache():
    """Reset the global data cache."""
    global_data['returns'] = None
    global_data['volume'] = None
    global_data['tickers'] = None

def load_data(tickers: list, start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple:
    """
    Load and cache stock data for given tickers and date range.

    Args:
        tickers (list): List of stock ticker symbols.
        start_date (pd.Timestamp): Start date for historical data.
        end_date (pd.Timestamp): End date for historical data.

    Returns:
        tuple: (returns DataFrame, volume Series, tickers list) or (None, None, None) if failed.
    """
    if global_data['returns'] is None:
        adj_close = pd.DataFrame()
        volumes = []
        for ticker in tickers:
            df = fetch_alpha_vantage_data(ticker)
            if not df.empty:
                adj_close[ticker] = df['4. close']
                volumes.append(df['5. volume'].mean())
            else:
                adj_close[ticker] = np.nan
        adj_close = adj_close[(adj_close.index >= start_date) & (adj_close.index <= end_date)]
        returns = adj_close.pct_change().dropna()
        volume = pd.Series(volumes, index=[ticker for ticker in tickers if ticker in adj_close.columns])
        if returns.empty:
            return None, None, None
        global_data['returns'] = returns
        global_data['volume'] = volume
        global_data['tickers'] = tickers
    return global_data['returns'], global_data['volume'], global_data['tickers']

def compute_mst_features(returns: pd.DataFrame, tickers: list, volume: pd.Series) -> tuple:
    """Compute MST-based features from returns data."""
    corr_matrix = returns.corr()
    G = nx.Graph()
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            corr = corr_matrix.iloc[i, j]
            weight = np.sqrt(2 * (1 - corr)) if not np.isnan(corr) else 0
            G.add_edge(tickers[i], tickers[j], weight=weight)
    mst = nx.minimum_spanning_tree(G, algorithm='kruskal')
    features = pd.DataFrame(index=tickers)
    features['degree'] = pd.Series(dict(mst.degree()))
    features['betweenness'] = pd.Series(nx.betweenness_centrality(mst))
    features['avg_edge_weight'] = pd.Series({
        node: np.mean([mst[node][neighbor]['weight'] for neighbor in mst.neighbors(node)]) if mst.degree(node) > 0 else 0
        for node in mst.nodes()
    })
    features['volatility'] = returns.std()
    features['avg_return'] = returns.mean()
    features['volume'] = volume
    features = features.fillna(0)
    return features, mst

def fit_skewed_normal_fmm(data: np.ndarray, n_components: int = 2, max_iter: int = 100) -> tuple:
    """Fit a skewed normal Finite Mixture Model to data."""
    np.random.seed(42)
    weights = np.ones(n_components) / n_components
    means = np.random.choice(data, n_components) if len(data) > 0 else np.zeros(n_components)
    scales = np.std(data) * np.ones(n_components) if len(data) > 0 else np.ones(n_components)
    shapes = np.zeros(n_components)
    for _ in range(max_iter):
        resp = np.zeros((len(data), n_components))
        for k in range(n_components):
            resp[:, k] = weights[k] * skewnorm.pdf(data, shapes[k], loc=means[k], scale=scales[k])
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp /= np.where(resp_sum > 0, resp_sum, 1)
        weights = resp.mean(axis=0)
        for k in range(n_components):
            sum_resp = resp[:, k].sum()
            if sum_resp > 0:
                means[k] = np.sum(resp[:, k] * data) / sum_resp
                scales[k] = np.sqrt(np.sum(resp[:, k] * (data - means[k])**2) / sum_resp)
                shapes[k] = np.sum(resp[:, k] * (data - means[k])**3) / (sum_resp * scales[k]**3)
            scales[k] = max(scales[k], 1e-6)
    return means, scales, shapes, weights

def select_best_distribution(data: np.ndarray) -> tuple:
    """Select the best distribution for data using KS test."""
    distributions = {
        'normal': norm,
        'student_t': t,
        'skew_normal': skewnorm
    }
    best_dist = None
    best_pvalue = -1
    best_params = None
    for name, dist in distributions.items():
        params = dist.fit(data)
        ks_stat, p_value = kstest(data, lambda x: dist.cdf(x, *params))
        if p_value > best_pvalue:
            best_pvalue = p_value
            best_dist = name
            best_params = params
    return best_dist, best_params, best_pvalue

def optimize_portfolio(returns: pd.DataFrame, risk_free_rate: float) -> np.ndarray:
    """Optimize portfolio weights using mean-variance optimization."""
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    num_assets = len(mean_returns)
    
    def negative_sharpe(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return - (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else np.inf
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array(num_assets * [1. / num_assets])
    result = minimize(negative_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x if result.success else initial_weights

def simulate_future_returns(model_type: str, historical_returns: np.ndarray, forecast_days: int, n_simulations: int = 1000, model_params: tuple = None) -> np.ndarray:
    """Simulate future returns using bootstrap or FMM."""
    simulated_paths = []
    for _ in range(n_simulations):
        if model_type == 'FMM':
            means, scales, shapes, weights = model_params
            component = np.random.choice(len(weights), p=weights)
            simulated_returns = skewnorm.rvs(a=shapes[component], loc=means[component], scale=scales[component], size=forecast_days)
        else:
            simulated_returns = np.random.choice(historical_returns, size=forecast_days, replace=True)
        simulated_paths.append(simulated_returns)
    return np.array(simulated_paths)

def compute_forecast_metrics(simulated_paths: np.ndarray, risk_free_rate: float) -> tuple:
    """Compute metrics from simulated return paths."""
    simulated_returns = np.mean(simulated_paths, axis=0)
    forecasted_return = np.mean(simulated_returns) * 252 * 100
    forecasted_vol = np.std(simulated_returns) * np.sqrt(252)
    forecasted_sharpe = (np.mean(simulated_returns) * 252 - risk_free_rate) / forecasted_vol if forecasted_vol > 0 else np.nan
    forecasted_var = np.quantile(simulated_returns, 0.05) * 100
    forecasted_cvar = np.mean(simulated_returns[simulated_returns <= np.quantile(simulated_returns, 0.05)]) * 100
    return forecasted_return, forecasted_sharpe, forecasted_var, forecasted_cvar

def kupiec_test(violations: int, n_observations: int, p: float = 0.05) -> tuple:
    """Perform Kupiec test for VaR back-testing."""
    if n_observations == 0 or violations < 0 or p <= 0 or p >= 1:
        return np.nan, np.nan
    # Avoid log(0) by adding small epsilon
    ratio = violations / n_observations if n_observations > 0 else p
    lr = -2 * np.log(((1 - p) ** (n_observations - violations) * p ** violations + 1e-10)) + 2 * np.log(((1 - ratio) ** (n_observations - violations) * ratio ** violations + 1e-10))
    p_value = 1 - chi2.cdf(lr, df=1)
    return lr, p_value

def back_test_var(returns: pd.Series, window_size: int = 100, step_size: int = 20, confidence: float = 0.95) -> tuple:
    """Back-test VaR on historical data using rolling windows."""
    if len(returns) <= window_size:
        return 0, 0, np.nan, np.nan  # Insufficient data
    violations = []
    for start in range(0, len(returns) - window_size, step_size):
        train = returns.iloc[start:start + window_size]
        test = returns.iloc[start + window_size:start + window_size + step_size]
        if len(test) == 0:
            continue
        var_pred = train.quantile(1 - confidence)
        violation = (test < var_pred).sum()
        violations.append(violation)
    total_violations = np.sum(violations)
    total_observations = len(returns) - window_size
    lr, p_value = kupiec_test(total_violations, total_observations, p=1 - confidence)
    return total_violations, total_observations, lr, p_value