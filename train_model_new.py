"""Clean training entry point for the portfolio risk model."""

from datetime import date
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

import model_utils

TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS"]
START_DATE = "2022-01-01"
END_DATE = date.today().isoformat()
ITERATIONS = 10_000
HORIZON = 30
SEED = 42


def main():
    prices = model_utils.fetch_data_yfinance(TICKERS, START_DATE, END_DATE)
    returns, mean_returns, covariance = model_utils.compute_returns_covariance(prices)
    weights = np.full(len(TICKERS), 1 / len(TICKERS))
    historical = returns @ weights
    var_95, cvar_95 = model_utils.compute_var_cvar(historical, 0.05)
    _, mc = model_utils.run_monte_carlo(prices.iloc[-1].to_numpy(), mean_returns.to_numpy(), covariance.to_numpy(), weights, HORIZON, ITERATIONS, SEED)

    model = {
        "metadata": {"created": date.today().isoformat(), "tickers": TICKERS, "start_date": START_DATE, "end_date": END_DATE},
        "model_params": {"mean_returns": mean_returns.to_dict(), "covariance_matrix": covariance.to_dict(), "weights": weights.tolist(), "latest_prices": prices.iloc[-1].tolist()},
        "performance": {"historical_var_95": float(var_95), "historical_cvar_95": float(cvar_95), "monte_carlo": {k: float(v) for k, v in mc.items()}},
    }
    model_utils.save_artifacts(model, StandardScaler().fit(prices), prices)
    Path("data").mkdir(exist_ok=True)
    Path("data/training_summary.json").write_text(json.dumps(model["performance"], indent=2), encoding="utf-8")
    print("Training complete ✓")


if __name__ == "__main__":
    main()
