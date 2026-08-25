# 📊 Stock Portfolio Risk Analysis

> **A probabilistic risk analytics dashboard for Indian equity portfolios.**
>
> Analyze historical risk, portfolio correlation and future uncertainty using **Monte Carlo simulation + Geometric Brownian Motion (GBM)**.

## ✨ What this project does

- 📈 Pulls market prices from **Yahoo Finance** for NSE stocks
- 🧮 Calculates **VaR, CVaR, volatility and loss probability**
- 🎲 Runs configurable **Monte Carlo portfolio simulations**
- 🧩 Visualizes **return correlations** between holdings
- 🔮 Generates probabilistic **future price forecasts** with 95% uncertainty bands
- 📥 Exports risk results as CSV
- 🎨 Provides a polished **Streamlit analytics dashboard**

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Dashboard | Streamlit |
| Data | yfinance, pandas, NumPy |
| Statistics | SciPy |
| Simulation | Monte Carlo, GBM |
| Visualization | Plotly |
| ML Utilities | scikit-learn, joblib |

## 📁 Project Structure

```text
Stock-Portfolio-Risk-Probability/
├── app.py                 # Streamlit dashboard
├── model_utils.py         # Data, risk and simulation utilities
├── train_model.py         # Reproducible training/artifact pipeline
├── requirements.txt       # Python dependencies
├── .gitignore             # Repository exclusions
├── data/                  # Generated market data / summaries
└── models/                # Generated model artifacts
```

> `data/` and `models/` artifacts are intentionally ignored by Git because they are generated locally or during deployment.

## 🚀 Run locally

```bash
git clone https://github.com/KAVYA-29-ai/Stock-Portfolio-Risk-Probability.git
cd Stock-Portfolio-Risk-Probability
pip install -r requirements.txt
streamlit run app.py
```

The dashboard fetches market data directly, so a pre-generated model file is **not required** just to explore the application.

### Optional: generate artifacts

```bash
python train_model.py
streamlit run app.py
```

## 📊 Risk Metrics

**Value at Risk (VaR)** estimates the loss threshold for a selected confidence level.

**Conditional Value at Risk (CVaR)** measures the average loss in the worst tail beyond VaR.

**Monte Carlo simulation** generates thousands of correlated portfolio outcomes from historical return and covariance estimates.

**GBM forecasting** produces probabilistic price paths rather than presenting a single deterministic prediction.

## ⚠️ Important Note

This project is an educational quantitative-finance application. Its simulations are based on historical statistical assumptions and **are not financial advice or guaranteed predictions of market performance**.

## 👤 Author

**KAVYA-29-ai**

Built as a hands-on project exploring **Python, quantitative risk analysis, financial modeling and data visualization**.

## 📄 License

MIT License
