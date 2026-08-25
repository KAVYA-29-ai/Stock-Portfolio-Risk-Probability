"""Interactive Streamlit dashboard for portfolio risk analysis."""

from datetime import date, datetime, timedelta
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import model_utils

DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS"]
DEFAULT_START = date.today() - timedelta(days=365 * 4)

st.set_page_config(page_title="Portfolio Risk Lab", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
[data-testid="stMetric"] {background: rgba(255,255,255,.035); border: 1px solid rgba(255,255,255,.10); padding: 18px; border-radius: 14px;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(255,255,255,.08);}
.hero {padding: 28px 30px; border: 1px solid rgba(255,255,255,.10); border-radius: 20px; background: linear-gradient(135deg, rgba(79,70,229,.18), rgba(14,165,233,.08)); margin-bottom: 22px;}
.hero h1 {margin: 0 0 8px 0; font-size: 2.25rem;}
.hero p {margin: 0; opacity: .75; font-size: 1rem;}
.section {font-size: 1.35rem; font-weight: 700; margin: 26px 0 12px;}
.small {opacity: .65; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)


def make_model(tickers, start_date, end_date):
    prices = model_utils.fetch_data_yfinance(tickers, start_date.isoformat(), (end_date + timedelta(days=1)).isoformat())
    returns, mean_returns, cov = model_utils.compute_returns_covariance(prices)
    return prices, returns, mean_returns, cov


def main():
    st.markdown('<div class="hero"><h1>📊 Portfolio Risk Lab</h1><p>Monte Carlo risk analysis, correlation intelligence and probabilistic price forecasting.</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Portfolio Controls")
        tickers = st.multiselect("Stocks", DEFAULT_TICKERS, default=DEFAULT_TICKERS)
        start_date = st.date_input("Start date", DEFAULT_START)
        end_date = st.date_input("End date", date.today())
        st.divider()
        iterations = st.slider("Monte Carlo simulations", 1_000, 20_000, 10_000, step=1_000)
        horizon = st.slider("Risk horizon (days)", 5, 365, 30)
        confidence = st.slider("Confidence level", 0.90, 0.99, 0.95, step=0.01)
        st.divider()
        equal_weights = st.checkbox("Equal portfolio weights", True)
        weights = []
        if tickers and not equal_weights:
            st.caption("Weights are normalized automatically.")
            for ticker in tickers:
                weights.append(st.number_input(ticker, min_value=0.0, max_value=1.0, value=1.0 / len(tickers), step=0.05))
            if sum(weights) == 0:
                st.error("At least one weight must be greater than 0.")
                return
            weights = np.asarray(weights, dtype=float)
            weights /= weights.sum()
        elif tickers:
            weights = np.ones(len(tickers), dtype=float) / len(tickers)

        refresh = st.button("🔄 Refresh market data", use_container_width=True)

    if not tickers:
        st.info("Select at least one stock from the sidebar to begin.")
        return
    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        return

    cache_key = (tuple(tickers), start_date, end_date, refresh)
    if "market_cache" not in st.session_state or st.session_state.get("cache_key") != cache_key:
        with st.spinner("Fetching market data and calculating risk inputs..."):
            prices, returns, mean_returns, cov = make_model(tickers, start_date, end_date)
        st.session_state.market_cache = (prices, returns, mean_returns, cov)
        st.session_state.cache_key = cache_key
    else:
        prices, returns, mean_returns, cov = st.session_state.market_cache

    latest_prices = prices.iloc[-1].to_numpy(dtype=float)
    _, mc = model_utils.run_monte_carlo(latest_prices, mean_returns.to_numpy(), cov.to_numpy(), weights, horizon, iterations)
    historical_var, historical_cvar = model_utils.compute_var_cvar((returns * weights).sum(axis=1), alpha=1 - confidence)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"VaR ({confidence:.0%})", f"{historical_var:.2%}", "Historical")
    c2.metric(f"CVaR ({confidence:.0%})", f"{historical_cvar:.2%}", "Historical")
    c3.metric(f"{horizon}D MC VaR", f"{mc['var_95']:.2%}")
    c4.metric("Probability of >10% loss", f"{mc['prob_loss_10']:.2%}")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "🧩 Correlation", "🎲 Monte Carlo", "🔮 Forecast"])

    with tab1:
        st.markdown('<div class="section">Historical Price Performance</div>', unsafe_allow_html=True)
        fig = px.line(prices, x=prices.index, y=tickers, template="plotly_dark")
        fig.update_layout(height=470, margin=dict(l=10, r=10, t=25, b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        returns_display = (prices.iloc[-1] / prices.iloc[0] - 1).sort_values(ascending=False)
        st.dataframe(pd.DataFrame({"Ticker": returns_display.index, "Total Return": returns_display.values}).style.format({"Total Return": "{:.2%}"}), use_container_width=True, hide_index=True)

    with tab2:
        st.markdown('<div class="section">Return Correlation Matrix</div>', unsafe_allow_html=True)
        corr = returns.corr()
        fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", template="plotly_dark")
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Higher positive correlation means stocks have historically tended to move together, reducing diversification benefits.")

    with tab3:
        st.markdown('<div class="section">Monte Carlo Loss Distribution</div>', unsafe_allow_html=True)
        simulated, mc = model_utils.run_monte_carlo(latest_prices, mean_returns.to_numpy(), cov.to_numpy(), weights, horizon, iterations)
        fig = go.Figure(go.Histogram(x=simulated, nbinsx=70, name="Simulated returns"))
        fig.add_vline(x=-mc["var_95"], line_dash="dash", annotation_text=f"95% VaR {mc['var_95']:.2%}")
        fig.update_layout(template="plotly_dark", height=460, xaxis_title="Portfolio return", yaxis_title="Scenarios", margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig, use_container_width=True)
        a, b, c = st.columns(3)
        a.metric("Expected return", f"{mc['mean']:.2%}")
        b.metric("Volatility", f"{mc['std']:.2%}")
        c.metric("CVaR", f"{mc['cvar_95']:.2%}")

        report = pd.DataFrame([{
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "stocks": ", ".join(tickers),
            "horizon_days": horizon,
            "simulations": iterations,
            "confidence": confidence,
            "historical_var": historical_var,
            "historical_cvar": historical_cvar,
            "monte_carlo_var": mc["var_95"],
            "monte_carlo_cvar": mc["cvar_95"],
            "probability_loss_10": mc["prob_loss_10"],
        }])
        st.download_button("⬇️ Download risk report", report.to_csv(index=False), "portfolio_risk_report.csv", "text/csv")

    with tab4:
        st.markdown('<div class="section">Probabilistic Price Forecast</div>', unsafe_allow_html=True)
        prediction_days = st.slider("Forecast horizon", 30, 365, 180, key="prediction_days")
        simulations = st.slider("Forecast paths", 100, 2_000, 1_000, step=100, key="forecast_simulations")
        if st.button("🚀 Generate forecast", type="primary"):
            with st.spinner("Running correlated GBM simulations..."):
                predictions = model_utils.predict_future_prices(latest_prices, mean_returns.to_numpy(), cov.to_numpy(), prediction_days, simulations)
            future_dates = pd.date_range(start=prices.index[-1], periods=prediction_days + 1, freq="B")
            for i, ticker in enumerate(tickers):
                history = prices[ticker].tail(252)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=history.index, y=history.values, name="Historical", mode="lines"))
                fig.add_trace(go.Scatter(x=future_dates, y=predictions["upper_95"][:, i], name="95% upper", line=dict(dash="dash")))
                fig.add_trace(go.Scatter(x=future_dates, y=predictions["lower_95"][:, i], name="95% lower", fill="tonexty", line=dict(dash="dash")))
                fig.add_trace(go.Scatter(x=future_dates, y=predictions["mean_path"][:, i], name="Mean forecast", mode="lines"))
                fig.update_layout(template="plotly_dark", title=ticker, height=430, hovermode="x unified", margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, use_container_width=True)
                final = predictions["mean_path"][-1, i]
                low = predictions["lower_95"][-1, i]
                high = predictions["upper_95"][-1, i]
                x, y, z = st.columns(3)
                x.metric("Expected price", f"₹{final:,.2f}", f"{(final/latest_prices[i]-1):.1%}")
                y.metric("95% lower", f"₹{low:,.2f}")
                z.metric("95% upper", f"₹{high:,.2f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        st.error(f"Dashboard error: {exc}")
        st.info("Try a wider date range or refresh the market data.")
