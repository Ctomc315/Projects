import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import date
import requests

api_key = os.getenv("GPqArrTegpdJ1qrA1zQKs7HhR47mam7t")
# ---------- 1. HELPER FUNCTION FOR POLYGON DATA ----------
def fetch_polygon_data(ticker, start_date, end_date, api_key):
    base_url = "https://api.polygon.io/v2/aggs/ticker"
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    url = (
        f"{base_url}/{ticker}/range/1/day/{start_str}/{end_str}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    )
    
    resp = requests.get(url)
    data = resp.json()
    
    if "results" not in data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data["results"])
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    
    df.rename(
        columns={
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume"
        },
        inplace=True
    )
    df["Adj Close"] = df["Close"]
    return df[["Open", "High", "Low", "Close", "Volume", "Adj Close"]]

# ---------- 2. STREAMLIT UI ----------
st.title('Polygon-Based Portfolio Dashboard')

# Ask the user for their Polygon API key
api_key = st.text_input("Enter your Polygon.io API Key:", type="password")
if not api_key:
    st.warning("Please enter a valid Polygon API Key.")
    st.stop()

# Tickers
assets_input = st.text_input(
    'Enter the assets you want to track (separated by commas):', 
    'AAPL, MSFT, GOOGL, AMZN, META'
)
tickers = [t.strip() for t in assets_input.split(',') if t.strip()]

# Start date
start_date = st.date_input(
    "Please enter your desired start date for the analysis:", 
    value=pd.to_datetime('2020-01-01')
)

# ---------- 3. FETCH DATA FOR TICKERS ----------
data_dict = {}
for ticker in tickers:
    df = fetch_polygon_data(ticker, start_date, date.today(), api_key)
    if df.empty:
        st.warning(f"No data returned for {ticker}. Check the ticker symbol or date range.")
    else:
        data_dict[ticker] = df["Adj Close"]  # We'll just store Adj Close in a dict

# Combine into a single DataFrame with each ticker as a column
if not data_dict:
    st.error("No valid data fetched for any ticker.")
    st.stop()
data = pd.DataFrame(data_dict)

st.write("### Adjusted Close Data", data)

# ---------- 4. PORTFOLIO CALCULATIONS ----------
ret_df = data.pct_change()
cumul_ret = (1 + ret_df).cumprod() - 1
pf_cumu_ret = cumul_ret.mean(axis=1)  # average across tickers

# Benchmark: use QQQ as a proxy for the S&P 500
benchmark_df = fetch_polygon_data("QQQ", start_date, date.today(), api_key)
if benchmark_df.empty:
    st.warning("No data returned for benchmark (QQQ).")
    bench_ret = pd.Series(dtype=float)
    bench_dev = pd.Series(dtype=float)
else:
    bench_ret = benchmark_df["Adj Close"].pct_change()
    bench_dev = (bench_ret + 1).cumprod() - 1

# Equal-weighted portfolio standard deviation
cov_matrix = ret_df.cov()
if cov_matrix.empty or len(cov_matrix) == 0:
    st.warning("Not enough data to calculate covariance.")
    pf_std = float('nan')
else:
    W = np.ones(len(cov_matrix)) / len(cov_matrix)
    pf_std = (W.dot(cov_matrix.dot(W))) ** 0.5

# Plot: Portfolio vs Benchmark
st.subheader('Portfolio Vs. Benchmark Development')
tog = pd.concat([bench_dev, pf_cumu_ret], axis=1)
tog.columns = ['QQQ (Benchmark)', 'Portfolio']
st.line_chart(tog)

# Portfolio Risk
st.subheader('Portfolio Risk')
st.write("Portfolio Std Dev:", pf_std)

# Benchmark Risk
st.subheader("Benchmark Risk")
bench_risk = bench_ret.std()
st.write("Benchmark Std Dev:", bench_risk)

# Portfolio Composition
st.subheader('Portfolio Composition')
fig, ax = plt.subplots(facecolor='#121212')
if len(tickers) > 1:
    ax.pie(W, labels=tickers, autopct='%1.1f%%', textprops={'color': 'white'})
else:
    ax.text(0.5, 0.5, f"Single Ticker: {tickers[0]}", 
            horizontalalignment='center', color='white')
st.pyplot(fig)

# Sharpe Ratio
st.subheader("Sharpe Ratio & Max Drawdown")
risk_free_rate = 0.0
daily_return = ret_df.mean(axis=1)
annualized_return = daily_return.mean() * 252
annualized_std = daily_return.std() * np.sqrt(252)
sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std

# Max Drawdown
cumulative_returns = (1 + daily_return).cumprod()
rolling_max = cumulative_returns.cummax()
drawdown = (cumulative_returns - rolling_max) / rolling_max
max_drawdown = drawdown.min()

st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")
st.write(f"**Maximum Drawdown:** {max_drawdown:.2%}")

#Sortino Ratio
# Assume 'ret_df' is your DataFrame with daily returns for each asset.
# For a portfolio, you might use the average daily return:
daily_return = ret_df.mean(axis=1)

# Set risk-free rate (annualized, e.g., 0% for simplicity)
risk_free_rate = 0.0

# Convert the risk-free rate to a daily rate (assuming 252 trading days)
daily_rf = risk_free_rate / 252

# Calculate the "downside" daily returns (returns below the risk-free rate)
downside_returns = daily_return[daily_return < daily_rf]

# Compute the downside deviation (annualized)
if len(downside_returns) > 0:
    downside_deviation = np.std(downside_returns) * np.sqrt(252)
else:
    downside_deviation = 0.0

# Annualized portfolio return (mean daily return annualized)
annualized_return = daily_return.mean() * 252

# Calculate the Sortino Ratio
if downside_deviation == 0:
    sortino_ratio = np.nan  # Avoid division by zero if there's no downside deviation
else:
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation

st.write(f"**Sortino Ratio:** {sortino_ratio:.2f}")

#Heatmap
import seaborn as sns
corr_matrix = ret_df.corr()
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

#EFFICIENT FRONTIER
def random_portfolios(returns, num_portfolios=5000):
    np.random.seed(42)
    results = np.zeros((3, num_portfolios))
    num_assets = returns.shape[1]
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0,i] = portfolio_std
        results[1,i] = portfolio_return
        results[2,i] = portfolio_return / portfolio_std  # Sharpe ratio
    return results

results = random_portfolios(ret_df)
# Plot
fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', marker='o')
ax.set_xlabel('Annualized Volatility')
ax.set_ylabel('Annualized Return')
ax.set_title('Efficient Frontier')
fig.colorbar(scatter, label='Sharpe Ratio')
st.pyplot(fig)