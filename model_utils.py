"""Core utilities for the Stock Portfolio Risk Analysis application."""

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

MODEL_DIR = Path("models")
DATA_DIR = Path("data")
MODEL_PATH = MODEL_DIR / "portfolio_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
PRICE_PATH = DATA_DIR / "portfolio_data.csv"


def fetch_data_yfinance(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted closing prices for the requested Yahoo Finance tickers."""
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        raise ValueError("Select at least one stock.")
    try:
        data = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="column")
    except Exception as exc:
        raise ValueError(f"Unable to fetch market data: {exc}") from exc
    if data.empty:
        raise ValueError("Yahoo Finance returned no data for the selected date range.")
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError("Close prices were not returned by Yahoo Finance.")
        prices = data["Close"].copy()
    else:
        if "Close" not in data.columns:
            raise ValueError("Close prices were not returned by Yahoo Finance.")
        prices = data[["Close"]].copy()
        prices.columns = [tickers[0]]
    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        raise ValueError(f"No price data found for: {', '.join(missing)}")
    prices = prices[tickers].apply(pd.to_numeric, errors="coerce")
    return prepare_portfolio_dataframe(prices)


def prepare_portfolio_dataframe(price_df: pd.DataFrame) -> pd.DataFrame:
    """Clean price data and remove rows that remain unusable."""
    clean = price_df.copy().replace([np.inf, -np.inf], np.nan)
    clean = clean.ffill().bfill().dropna(how="all").dropna(axis=1, how="all")
    if clean.empty:
        raise ValueError("The price dataset is empty after cleaning.")
    return clean


def compute_returns_covariance(price_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Compute daily log returns, mean returns and covariance matrix."""
    log_returns = np.log(price_df / price_df.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    if log_returns.empty:
        raise ValueError("Not enough historical observations to calculate returns.")
    return log_returns, log_returns.mean(), log_returns.cov()


def _stable_cholesky(cov_matrix: np.ndarray) -> np.ndarray:
    """Return a Cholesky factor, adding tiny diagonal jitter when needed."""
    cov = np.asarray(cov_matrix, dtype=float)
    cov = (cov + cov.T) / 2
    jitter = 1e-12
    for _ in range(8):
        try:
            return np.linalg.cholesky(cov + np.eye(cov.shape[0]) * jitter)
        except np.linalg.LinAlgError:
            jitter *= 10
    raise ValueError("The covariance matrix is not positive semi-definite.")


def compute_var_cvar(portfolio_returns: pd.Series, alpha: float = 0.05) -> Tuple[float, float]:
    """Compute positive-loss VaR and CVaR at the requested tail probability."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    returns = pd.Series(portfolio_returns).dropna().to_numpy(dtype=float)
    if returns.size == 0:
        raise ValueError("No portfolio returns available for risk calculation.")
    var = float(-np.quantile(returns, alpha))
    tail = returns[returns <= -var]
    return var, float(-tail.mean()) if tail.size else var


def run_monte_carlo(S0: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray, weights: np.ndarray,
                    days: int = 30, iterations: int = 10_000, random_state: int = 42) -> Tuple[np.ndarray, Dict[str, float]]:
    """Simulate correlated GBM portfolio returns and calculate risk statistics."""
    if days < 1 or iterations < 100:
        raise ValueError("days must be >= 1 and iterations must be >= 100.")
    prices = np.asarray(S0, dtype=float)
    mu = np.asarray(mean_returns, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if prices.ndim != 1 or mu.shape != prices.shape or weights.shape != prices.shape:
        raise ValueError("S0, mean_returns and weights must have matching shapes.")
    if weights.sum() <= 0:
        raise ValueError("Portfolio weights must contain a positive total.")
    weights = weights / weights.sum()
    chol = _stable_cholesky(np.asarray(cov_matrix, dtype=float))
    rng = np.random.default_rng(random_state)
    z = rng.standard_normal((iterations, days, len(prices)))
    cumulative_log_returns = (mu + z @ chol.T).sum(axis=1)
    final_prices = prices * np.exp(cumulative_log_returns)
    initial_value = float(prices @ weights)
    portfolio_returns = (final_prices @ weights) / initial_value - 1.0
    var_95 = float(-np.quantile(portfolio_returns, 0.05))
    tail = portfolio_returns[portfolio_returns <= -var_95]
    stats = {
        "mean": float(portfolio_returns.mean()),
        "std": float(portfolio_returns.std()),
        "var_95": var_95,
        "cvar_95": float(-tail.mean()) if tail.size else var_95,
        "prob_loss_10": float(np.mean(portfolio_returns < -0.10)),
    }
    return portfolio_returns, stats


def predict_future_prices(latest_prices: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray,
                          days: int = 252, simulations: int = 1_000, random_state: int = 42,
                          annualization_factor: float = 252) -> Dict[str, np.ndarray]:
    """Generate correlated GBM price paths with 5th and 95th percentile bands."""
    if days < 1 or simulations < 100:
        raise ValueError("days must be >= 1 and simulations must be >= 100.")
    latest = np.asarray(latest_prices, dtype=float)
    mu = np.asarray(mean_returns, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    n_stocks = len(latest)
    if mu.shape != (n_stocks,) or cov.shape != (n_stocks, n_stocks):
        raise ValueError("Prediction inputs have incompatible shapes.")
    chol = _stable_cholesky(cov)
    rng = np.random.default_rng(random_state)
    dt = 1.0 / annualization_factor
    daily_drift = mu - 0.5 * np.diag(cov)
    prices = np.empty((simulations, days + 1, n_stocks), dtype=float)
    prices[:, 0, :] = latest
    for day in range(1, days + 1):
        shocks = rng.standard_normal((simulations, n_stocks)) @ chol.T
        step = np.exp(daily_drift * dt + shocks * np.sqrt(dt))
        prices[:, day, :] = prices[:, day - 1, :] * step
    return {
        "mean_path": prices.mean(axis=0),
        "upper_95": np.percentile(prices, 95, axis=0),
        "lower_95": np.percentile(prices, 5, axis=0),
    }


def save_artifacts(model_obj: Dict, scaler: object, price_df: pd.DataFrame) -> None:
    """Persist model metadata and historical prices."""
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    joblib.dump(model_obj, MODEL_PATH)
    if scaler is not None:
        joblib.dump(scaler, SCALER_PATH)
    price_df.to_csv(PRICE_PATH)


def load_artifacts() -> Tuple[Dict, object, pd.DataFrame]:
    """Load saved artifacts and historical prices."""
    missing = [str(path) for path in (MODEL_PATH, PRICE_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing artifacts: " + ", ".join(missing))
    model_obj = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    price_df = pd.read_csv(PRICE_PATH, index_col=0, parse_dates=True)
    return model_obj, scaler, prepare_portfolio_dataframe(price_df)
