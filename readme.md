# Reinforcement Learning for Optimal Trade Execution in High-Frequency Markets

This repository contains a reproducible prototype of an algorithmic trading framework that blends inventory theory (EOQ) with machine learning (LSTM + XGBoost), plus ABC analysis and Bayesian-style configuration variants, to size trades adaptively and evaluate performance at the portfolio level. The design mirrors the accompanying research write-up “Inventory Stock” and its experimental protocol and KPIs.

## What this project does

- Fetches and prepares data for a mixed-asset universe (US equities, FX pairs, and commodities) using Alpha Vantage.
- Engineers technical features (MA5/10/20, RSI, MACD, 5-day return, 20-day volatility).
- Selects predictive features per asset via XGBoost.
- Forecasts next-day prices with a multivariate LSTM using the selected features.
- Maps assets to A/B/C tiers by historical contribution (Pareto-style).
- Translates forecasts into “demand” and sizes orders using an EOQ rule; ABC tiers modulate thresholds and holding costs.
- Backtests four strategy variants (LSTM only, +ABC, +EOQ, Full System) and reports Cumulative Return, Sharpe, Max Drawdown, Total Transaction Cost.
- Outputs a portfolio summary CSV with all metrics. 

## Project contents

- **main.py** — End-to-end pipeline: data ingestion, features, modeling, ABC/EOQ logic, simulation, and reporting. Running it produces `portfolio_backtest_summary.csv`. 
- **Research PDF** — Conceptual background, methodology, KPIs, and example results used to guide the implementation.

## Requirements

- Python 3.9–3.11
- Packages (install via `pip install -r requirements.txt` if you create one):
  - pandas, numpy, matplotlib
  - alpha_vantage
  - scikit-learn
  - xgboost
  - tensorflow (for Keras / LSTM)
- An Alpha Vantage API key

** Security note:** The example code currently hardcodes an API key string. Use your own key via environment variables and avoid committing secrets. See “Configuration” below. 

##  Configuration

Set your Alpha Vantage key before running:

```bash
export ALPHAVANTAGE_API_KEY="YOUR_KEY"
```

In `main.py`, replace the hardcoded key with:

```python
api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
```

(The script already imports `os`). 

## ▶️ How to run

```bash
python main.py
```

### What you’ll see/produce:

**Console:** progress prints and a final DataFrame with portfolio metrics for each variant.  
**Files:**  
- `asset_data/*.csv` — cached, preprocessed time series per asset (auto-created).  
- `portfolio_backtest_summary.csv` — one row per strategy variant with metrics. 

## 🧩 Code walkthrough (what each part does)

### 1) Asset universe & data access
- Assets list includes stocks (AAPL, MSFT, TSLA, AMZN), FX (EURUSD, USDJPY), and commodities (COPPER, NATURAL_GAS).
- `load_or_fetch_data(asset, api_key, directory="asset_data")`: caches per symbol.
- `prepare_data(data)`: harmonizes API outputs into OHLCV schema.

### 2) Feature engineering
- `add_technical_indicators(df)` creates MA5, MA10, MA20, Return_5, Volatility_20, RSI, MACD.

### 3) Feature selection (XGBoost)
- `select_features(df, target_col="close", symbol="")`: uses XGBRegressor, keeps top 5 features.

### 4) Forecasting (LSTM)
- `lstm_forecast_multivariate(data, target="close")`: multivariate LSTM with scaling, windowing, training.

### 5) ABC classification
- `classify_stocks_by_return(stock_stats_df)`: tags A (top 20%), B (next 30%), C (remaining).

### 6) EOQ trade sizing & simulation
- `calculate_eoq(D, S, H)`: standard EOQ formula.
- `simulate_portfolio(...)`: runs backtest with ABC, EOQ, and thresholds.

### 7) Experiments driver
- `run_portfolio_experiments(data_dict, abc_map)`: runs four variants — LSTM Only, +ABC, +EOQ, Full System.

## 🗂️ Data & caching

- First run downloads full histories from Alpha Vantage and writes per-symbol CSVs.  
- Subsequent runs load from cache to avoid rate limits. 

## 📈 Outputs

- `portfolio_backtest_summary.csv` — Portfolio-level metrics per strategy variant.  
- Optional matplotlib plots for feature importance (commented out). 

## 🔧 Troubleshooting

- **API rate limits / empty frames**: use cached CSVs.  
- **Shape mismatches in LSTM**: check rolling window drops.  
- **Secrets in code**: replace hardcoded API key with env variable. 

## 🧪 Reproducibility notes

- LSTM uses small epoch count, no fixed seeds (run-to-run variation).  
- Commodity series are monthly — consider resampling others for alignment. 

## 🧭 Suggested extensions

- Config file (YAML/JSON) for assets and hyperparameters.  
- Bayesian optimization (e.g., Optuna).  
- Risk controls, better transaction cost modeling.  
- Unified resampling for mixed-frequency assets. 

## 📚 Background / Citation

If you use or extend this codebase in academic or industrial work, please cite the research write-up *Inventory Stock* and reference the implementation (`main.py`).

## 📝 License

Add your preferred license (e.g., MIT) here.

---

## Quick start TL;DR

```bash
# 1) Install dependencies
pip install pandas numpy matplotlib alpha_vantage scikit-learn xgboost tensorflow

# 2) Set your key
export ALPHAVANTAGE_API_KEY="YOUR_KEY"

# 3) Run
python main.py

# 4) Inspect results
cat portfolio_backtest_summary.csv
```
