import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.commodities import Commodities

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import Input
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

import os
from datetime import datetime
import pickle
from pathlib import Path
from keras.models import load_model

MODEL_DIR = Path("models/lstm")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LSTM_WINDOW = 60  # single source of truth
BACKTEST_YEARS = 2  # enforce last-2-years window for the backtest

results = {}
stock_stats = []
GLOBAL_START = datetime(2010, 6, 29)

# api_key = "GLVZ9GJN4IW7GRUB"
api_key = "K757OWEW19L34ML9"
# symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']  # Multiple stocks for analysis
ts = TimeSeries(key=api_key, output_format="pandas")
fx = ForeignExchange(key=api_key, output_format="pandas")

assets = [
    {"symbol": "AAPL", "type": "stock"},
    {"symbol": "MSFT", "type": "stock"},
    {"symbol": "TSLA", "type": "stock"},
    {"symbol": "AMZN", "type": "stock"},
    {"symbol": "EURUSD", "type": "forex", "from_symbol": "EUR", "to_symbol": "USD"},
    {"symbol": "USDJPY", "type": "forex", "from_symbol": "USD", "to_symbol": "JPY"},
]


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # drop accidental index columns from prior saves
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    cols_lower = [c.lower() for c in df.columns]

    # Case 1: Stock/forex style (AV or your yfinance-normalized CSVs)
    if "1. open" in df.columns:
        df = df.rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume",
            }
        )[["open", "high", "low", "close", "volume"]]

    # Case 2: Commodity style -> timestamp,value (raw AV commodity pull)
    elif "timestamp" in cols_lower and "value" in cols_lower:
        rename_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl == "timestamp":
                rename_map[c] = "date"
            elif cl == "value":
                rename_map[c] = "close"
        df = df.rename(columns=rename_map)

        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["volume"] = 0
        df = df[["open", "high", "low", "close", "volume"]]

    # Case 3: Generic single-value CSV with 'value'
    elif "value" in df.columns:
        df = df.rename(columns={"value": "close"})
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["volume"] = 0
        df = df[["open", "high", "low", "close", "volume"]]

    else:
        # Fallback: ensure 'close' exists
        if "close" not in df.columns:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                raise ValueError(
                    f"Unknown data format, cannot infer 'close' from: {df.columns.tolist()}"
                )
            df["close"] = df[num_cols[-1]]
        df["open"] = df.get("open", df["close"])
        df["high"] = df.get("high", df["close"])
        df["low"] = df.get("low", df["close"])
        df["volume"] = df.get("volume", 0)
        df = df[["open", "high", "low", "close", "volume"]]

    # Ensure oldest → newest for modeling
    df = df[::-1].reset_index(drop=True)

    # ✅ enforce global start date for all assets
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"] >= GLOBAL_START]
        df = df.reset_index(drop=True)

    return df


def load_or_fetch_data(asset, api_key, directory="asset_data"):
    os.makedirs(directory, exist_ok=True)
    symbol = asset["symbol"]
    filepath = os.path.join(directory, f"{symbol}.csv")

    # Symbols you manage locally via yfinance (do NOT refetch/overwrite)
    LOCAL_ONLY_SYMBOLS = {"COPPER", "NATURAL_GAS"}

    # If a cached CSV exists, always use it
    if os.path.exists(filepath):
        print(f"Loading cached data for {symbol}")
        df = pd.read_csv(filepath)
        df = prepare_data(df)
        return df

    # If it's a local-only symbol and no file exists yet, fail loudly with guidance
    if symbol in LOCAL_ONLY_SYMBOLS:
        raise FileNotFoundError(
            f"{symbol}.csv not found in '{directory}'. "
            f"Generate it first with the yfinance script, saved as "
            f"'date,1. open,2. high,3. low,4. close,5. volume'."
        )

    # Otherwise, fetch from Alpha Vantage for stocks/forex
    print(f"Fetching data for {symbol} from Alpha Vantage...")

    if asset["type"] == "stock":
        ts = TimeSeries(key=api_key, output_format="pandas")
        df_raw, _ = ts.get_daily(symbol=symbol, outputsize="full")
        # print first 5 rows with the headers
        # print(df_raw.head())

    elif asset["type"] == "forex":
        fx = ForeignExchange(key=api_key, output_format="pandas")
        df_raw, _ = fx.get_currency_exchange_daily(
            from_symbol=asset["from_symbol"],
            to_symbol=asset["to_symbol"],
            outputsize="full",
        )
        # AV doesn't provide FX volume
        df_raw["5. volume"] = 0
    else:
        raise ValueError(f"Unknown asset type {asset['type']}")

    # Make sure we're working on a fresh copy
    df_raw = df_raw.copy()
    # Turn the index into a column
    df_raw = df_raw.reset_index()
    # Rename the index column explicitly
    df_raw = df_raw.rename(columns={"index": "date"})

    # print(df_raw.head())

    # # ✅ Save cleanly (no index) so reloads don't create duplicate 'date' columns
    df_raw.to_csv(filepath, index=False)
    print(f"Saved {symbol} data to {filepath}")

    # Normalize columns and order
    df = prepare_data(df_raw)
    return df


def last_two_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the last BACKTEST_YEARS of data by 'date' column if present.
    Assumes df is oldest→newest after prepare_data().
    """
    if "date" in df.columns:
        end = pd.to_datetime(df["date"]).max()
        start = end - pd.Timedelta(days=365 * BACKTEST_YEARS)
        out = df[pd.to_datetime(df["date"]).between(start, end)].reset_index(drop=True)
        return out
    return df


def _artifact_paths(symbol: str):
    model_path = MODEL_DIR / f"{symbol}.keras"
    meta_path = MODEL_DIR / f"{symbol}_meta.pkl"
    return model_path, meta_path


def _save_meta(meta_path: Path, meta: dict):
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)


def _load_meta(meta_path: Path) -> dict:
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def add_technical_indicators(df):
    if "close" not in df.columns:
        raise KeyError("The 'close' column is missing in the DataFrame.")

    df["MA5"] = df["close"].rolling(window=5).mean()
    df["MA10"] = df["close"].rolling(window=10).mean()
    df["MA20"] = df["close"].rolling(window=20).mean()
    df["Return_5"] = df["close"].pct_change(periods=5)
    df["Volatility_20"] = df["close"].rolling(window=20).std()
    df["RSI"] = compute_rsi(df["close"], 14)
    df["MACD"] = compute_macd(df["close"])
    df = df.dropna()
    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, slow=26, fast=12):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def plot_xgboost_feature_importance(model, feature_names, symbol):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(
        range(len(feature_names)), [feature_names[i] for i in indices], rotation=45
    )
    plt.title(f"{symbol} - XGBoost Feature Importances")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.show()


def select_features(df, target_col="close", symbol=""):
    df = df.select_dtypes(include=[np.number])

    # target: future return (use percent change)
    y = df[target_col].pct_change().shift(-1).dropna()
    df = df.iloc[:-1].reset_index(drop=True)  # Align features to y

    X = df.drop(columns=[target_col])
    model = XGBRegressor(random_state=42)  # setting a seed for reproducibility
    model.fit(X, y)

    # plot_xgboost_feature_importance(model, X.columns, symbol)

    importances = model.feature_importances_
    feature_importance_df = (
        pd.DataFrame({"Feature": X.columns, "Importance": importances})
        .sort_values(by="Feature")
        .reset_index(drop=True)
    )

    print("\nFull Feature Importance Ranking:")
    print(feature_importance_df)

    top_features = X.columns[np.argsort(model.feature_importances_)][-5:]
    X_selected = X[top_features].copy().reset_index(drop=True)
    y_target = (
        df[target_col].iloc[1:].reset_index(drop=True)
    )  # align with prediction target

    y_target.name = target_col  # ✅ set the name to 'Close' (important for merging)

    return X_selected, y_target


def tune_xgb_hyperparams(X, y, n_iter=25, cv=3, random_state=42):
    """
    Lightweight hyperparam search for XGBRegressor.
    Returns: best_estimator_, best_params_
    """
    base = XGBRegressor(
        n_estimators=600,
        tree_method="hist",  # fast CPU training
        random_state=random_state,
    )
    param_dist = {
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7, 10],
        "gamma": [0, 0.1, 0.2, 0.3],
        "reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0],
        "reg_alpha": [0.0, 0.5, 1.0, 2.0],
    }
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=0,
        random_state=random_state,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def lstm_forecast_multivariate(
    data: pd.DataFrame,
    target: str = "close",
    symbol: str | None = None,
    use_cache: bool = True,
    force_retrain: bool = False,
):
    """
    Train (or load) an LSTM for one asset and return (actual, pred) arrays on the test split.
    Caches: Keras model + fitted MinMaxScaler + feature schema under models/lstm/.
    """
    assert isinstance(target, str), "Target must be a string"
    assert target in data.columns, f"Target column '{target}' not found"

    feature_cols = [c for c in data.columns if c != target]
    full_data = data[feature_cols + [target]]  # fixed order
    target_index = full_data.columns.get_loc(target)

    model_path = meta_path = None
    if symbol:
        model_path, meta_path = _artifact_paths(symbol)

    # ---------- Try cache ----------
    if (
        use_cache
        and not force_retrain
        and symbol is not None
        and model_path.exists()
        and meta_path.exists()
    ):
        try:
            meta = _load_meta(meta_path)
            schema_ok = (
                meta.get("target") == target
                and meta.get("feature_cols") == feature_cols
                and meta.get("window") == LSTM_WINDOW
            )
            if schema_ok:
                model = load_model(model_path)
                scaler = meta["scaler"]

                scaled = scaler.transform(full_data)
                X_all, y_all = [], []
                for i in range(LSTM_WINDOW, len(scaled)):
                    X_all.append(scaled[i - LSTM_WINDOW : i, :-1])
                    y_all.append(scaled[i, target_index])
                X_all, y_all = np.array(X_all), np.array(y_all)

                # ordered time split: last 20% = test
                split = int(len(X_all) * 0.8)
                X_test, y_test = X_all[split:], y_all[split:]

                pred_scaled = model.predict(X_test, verbose=0)
                y_test = y_test.reshape(-1, 1)

                pred_full = np.zeros((len(pred_scaled), full_data.shape[1]))
                actual_full = np.zeros_like(pred_full)
                pred_full[:, target_index] = pred_scaled[:, 0]
                actual_full[:, target_index] = y_test[:, 0]

                # pred = MinMaxScaler().inverse_transform(
                #     np.where(
                #         np.arange(full_data.shape[1]) == target_index, pred_full, 0
                #     )
                # )  # not used; see below
                # # The above is not correct; inverse needs the *same* scaler. Build properly:
                pred = scaler.inverse_transform(pred_full)[:, target_index]
                actual = scaler.inverse_transform(actual_full)[:, target_index]
                return actual, pred
            else:
                print(f"[{symbol}] Cached model schema mismatch → retraining...")
        except Exception as e:
            print(f"[{symbol}] Failed to load cached model: {e}. Retraining...")

    # ---------- Train fresh ----------
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(full_data)

    X_all, y_all = [], []
    for i in range(LSTM_WINDOW, len(scaled)):
        X_all.append(scaled[i - LSTM_WINDOW : i, :-1])
        y_all.append(scaled[i, target_index])
    X_all, y_all = np.array(X_all), np.array(y_all)

    # ordered time split: last 20% = test
    split = int(len(X_all) * 0.8)
    X_train, y_train = X_all[:split], y_all[:split]
    X_test, y_test = X_all[split:], y_all[split:]

    model = Sequential(
        [
            Input(shape=(X_all.shape[1], X_all.shape[2])),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    pred_scaled = model.predict(X_test, verbose=0)
    y_test = y_test.reshape(-1, 1)

    pred_full = np.zeros((len(pred_scaled), full_data.shape[1]))
    actual_full = np.zeros_like(pred_full)
    pred_full[:, target_index] = pred_scaled[:, 0]
    actual_full[:, target_index] = y_test[:, 0]

    pred = scaler.inverse_transform(pred_full)[:, target_index]
    actual = scaler.inverse_transform(actual_full)[:, target_index]

    # Save artifacts
    if symbol and use_cache:
        try:
            model.save(model_path)
            meta = {
                "scaler": scaler,
                "feature_cols": feature_cols,
                "target": target,
                "window": LSTM_WINDOW,
                "trained_at": datetime.utcnow().isoformat(),
            }
            _save_meta(meta_path, meta)
            print(f"[{symbol}] Saved model → {model_path}")
            print(f"[{symbol}] Saved meta   → {meta_path}")
        except Exception as e:
            print(f"[{symbol}] Warning: failed to save artifacts: {e}")

    return actual, pred


def classify_stocks_by_return(stock_stats_df):
    # Sort by cumulative return descending
    sorted_df = stock_stats_df.sort_values(
        by="cumulative_return", ascending=False
    ).reset_index(drop=True)
    n = len(sorted_df)
    a_cutoff = int(0.2 * n)
    b_cutoff = int(0.5 * n)

    categories = [
        "A" if i < a_cutoff else "B" if i < b_cutoff else "C" for i in range(n)
    ]
    sorted_df["ABC"] = categories
    return sorted_df


def calculate_eoq(D, S, H):
    return math.sqrt((2 * D * S) / H) if H > 0 else 50


def simulate_portfolio(assets_data: dict, abc_map: dict, variant_config: dict):
    """
    assets_data: {symbol: {"actual": np.array, "pred": np.array, "vol": np.array}}
    abc_map:     {symbol: "A"/"B"/"C"}
    variant_config: {"use_abc": bool, "use_eoq": bool, "S": float, "H": 0 or "vol", "tuned": bool}
    """
    cash = 100000.0
    holdings = {symbol: 0 for symbol in assets_data}
    total_cost = 0.0
    portfolio_values = []

    S = float(variant_config.get("S", 10.0))
    H = variant_config.get("H", 0)  # 0 or "vol"
    use_eoq = bool(variant_config.get("use_eoq", False))
    use_abc = bool(variant_config.get("use_abc", False))

    # ---- timeline sync: truncate to common min length ----
    min_len = min(len(v["actual"]) for v in assets_data.values())

    for t in range(min_len):
        daily_value = cash
        for symbol, data in assets_data.items():
            price = data["actual"][t]
            pred = data["pred"][t]
            signal = pred - price

            # ABC gating
            abc = abc_map.get(symbol, "B")
            threshold = 0.02 if (use_abc and abc == "C") else 0.01
            if abs(signal) < threshold:
                daily_value += holdings[symbol] * price
                continue

            # Demand proxy D from forecast magnitude
            D = abs(signal) * 100

            # Holding cost H_adj: if "vol", scale by rolling vol; else pass-through
            if use_eoq:
                if H == "vol":
                    vol_t = float(data.get("vol", np.zeros(min_len))[t])
                    # scale vol to a per-trade holding cost (simple, consistent proxy)
                    H_adj = max(1e-6, 100.0 * vol_t)
                else:
                    H_adj = float(H)
                size = int(calculate_eoq(D, S, H_adj))
            else:
                size = 80 if abc == "A" else 50 if abc == "B" else 30

            if size <= 0:
                daily_value += holdings[symbol] * price
                continue

            # Execute
            if signal > 0 and cash >= size * price + S:  # Buy
                holdings[symbol] += size
                cash -= size * price + S
                total_cost += S
            elif signal < 0 and holdings[symbol] >= size:  # Sell
                holdings[symbol] -= size
                cash += size * price - S
                total_cost += S

            daily_value += holdings[symbol] * price

        portfolio_values.append(daily_value)

    return portfolio_values, total_cost


def compute_portfolio_metrics(values, forecast_rmse):
    values = np.array(values)
    returns = np.diff(values) / values[:-1]
    cumulative_return = (values[-1] - values[0]) / values[0]
    sharpe = np.mean(returns) / np.std(returns) if np.std(returns) else 0
    drawdown = np.max(np.maximum.accumulate(values) - values)
    max_drawdown = drawdown / np.max(np.maximum.accumulate(values))
    return {
        "Cumulative Return": cumulative_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
        "Forecast RMSE": forecast_rmse,
    }


def run_portfolio_experiments(data_dict: dict, abc_map_seed: dict | None = None):
    """
    data_dict: {symbol: raw_df from load_or_fetch_data()}
    Returns: DataFrame summarizing portfolio metrics by configuration and writes CSVs.
    """
    results = []
    per_asset_rows = []

    # === Configs to match paper ===
    configs = {
        "LSTM Only": {
            "use_abc": False,
            "use_eoq": False,
            "S": 10.0,
            "H": 0.0,
            "tuned": False,
        },
        "LSTM + ABC": {
            "use_abc": True,
            "use_eoq": False,
            "S": 10.0,
            "H": 0.0,
            "tuned": False,
        },
        "ABC + EOQ": {
            "use_abc": True,
            "use_eoq": True,
            "S": 10.0,
            "H": "vol",
            "tuned": False,
        },
        "Tuned ABC + EOQ": {
            "use_abc": True,
            "use_eoq": True,
            "S": 10.0,
            "H": "vol",
            "tuned": True,
        },
    }

    # === Build forecasts & metrics per asset (last 2 years, ordered split) ===
    assets_data = {}
    for symbol, raw_df in data_dict.items():
        try:
            df = last_two_years(raw_df)
            df = add_technical_indicators(df)
            X_sel, y_target = select_features(df, symbol=symbol)
            merged = (
                pd.concat([X_sel, y_target], axis=1).dropna().reset_index(drop=True)
            )

            actual, pred = lstm_forecast_multivariate(
                merged, target="close", symbol=symbol
            )

            # Per-asset metrics
            rmse = float(math.sqrt(mean_squared_error(actual, pred)))
            mae = float(mean_absolute_error(actual, pred))

            # Rolling vol on actual returns (window=20), aligned to test portion length
            actual_series = pd.Series(actual)
            ret = actual_series.pct_change().fillna(0.0).values
            vol = (
                pd.Series(ret)
                .rolling(20)
                .std()
                .fillna(method="bfill")
                .fillna(0.0)
                .values
            )

            assets_data[symbol] = {
                "actual": np.asarray(actual),
                "pred": np.asarray(pred),
                "vol": vol,
            }

            per_asset_rows.append(
                {
                    "symbol": symbol,
                    "RMSE": rmse,
                    "MAE": mae,
                    "test_len": len(actual),
                }
            )
        except Exception as e:
            print(f"[{symbol}] Skipped due to error: {e}")

    if not assets_data:
        raise RuntimeError("No assets produced forecasts — aborting.")

    # === ABC map (from per-asset test cumulative return) ===
    stock_stats = []
    for symbol, v in assets_data.items():
        prices = np.asarray(v["actual"])
        if len(prices) < 2:
            cumret = 0.0
        else:
            cumret = (prices[-1] - prices[0]) / prices[0]
        stock_stats.append({"symbol": symbol, "cumulative_return": float(cumret)})

    stock_stats_df = pd.DataFrame(stock_stats)
    classified_df = classify_stocks_by_return(stock_stats_df)
    abc_map = dict(zip(classified_df["symbol"], classified_df["ABC"]))
    if abc_map_seed:
        # allow seed/overrides from outside, if passed
        abc_map.update(abc_map_seed)

    print(classified_df)

    # === Portfolio runs by configuration ===
    # for fair comparison, compute average RMSE (not used in trading)
    avg_rmse = float(np.mean([r["RMSE"] for r in per_asset_rows]))

    for name, config in configs.items():
        portfolio_vals, total_cost = simulate_portfolio(
            assets_data, abc_map=abc_map, variant_config=config
        )
        metrics = compute_portfolio_metrics(portfolio_vals, avg_rmse)
        metrics["Model Variant"] = name
        metrics["Total Transaction Cost"] = total_cost
        results.append(metrics)

    results_df = pd.DataFrame(results)

    # === Persist outputs ===
    Path("outputs").mkdir(exist_ok=True, parents=True)
    pd.DataFrame(per_asset_rows).to_csv(
        "outputs/forecast_metrics_by_asset.csv", index=False
    )
    results_df.to_csv("outputs/portfolio_backtest_summary.csv", index=False)
    classified_df.to_csv("outputs/abc_classification.csv", index=False)

    print("Saved outputs to outputs/")

    return results_df


# data = load_or_fetch_data(symbols[4], api_key)
data_dict = {}
for asset in assets:
    df = load_or_fetch_data(asset, api_key)
    data_dict[asset["symbol"]] = df

df.head()  # Display the first few rows of the data

final_results_df = run_portfolio_experiments(data_dict)
print(final_results_df)
