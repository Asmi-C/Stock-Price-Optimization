# import requests
# import pandas as pd
# from io import StringIO

# API_KEY = "K757OWEW19L34ML9"

# url = f"https://www.alphavantage.co/query?function=WTI&interval=monthly&datatype=csv&apikey={API_KEY}"
# resp = requests.get(url)

# df = pd.read_csv(StringIO(resp.text))
# print(df.head())


# import requests
# import pandas as pd
# from io import StringIO

# API_KEY = "YOUR_KEY"  # <-- replace with your Alpha Vantage key

# url = f"https://www.alphavantage.co/query?function=COPPER&interval=daily&datatype=csv&apikey={API_KEY}"
# resp = requests.get(url)
# resp.raise_for_status()

# df = pd.read_csv(StringIO(resp.text))
# print(df.head(10))  # show first 10 rows


# import yfinance as yf

# # COMEX Copper Futures
# ticker = yf.Ticker("HG=F")
# df = ticker.history(period="6mo", interval="1d")

# print(df.head())

# # 2006-08-01, 1999-11-01, 1999-11-01, 1999-11-01, 2010-06-29, 2006-08-01, 1970-01-01

# fetch_commodities_yf.py
import os
from datetime import date
import pandas as pd
import yfinance as yf

OUT_DIR = "asset_data"
START_DATE = "2010-06-29"  # align with TSLA availability
END_DATE = None  # None = up to today

os.makedirs(OUT_DIR, exist_ok=True)


def fetch_and_save_yf(
    ticker: str, output_symbol: str, start: str = START_DATE, end: str | None = END_DATE
):
    """
    Fetch daily OHLCV from Yahoo Finance and save as:
    date,1. open,2. high,3. low,4. close,5. volume
    In reverse chronological order (today → oldest) to match Alpha Vantage CSVs.
    """
    df = yf.Ticker(ticker).history(
        start=start, end=end, interval="1d", auto_adjust=False
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check symbol or date range.")

    # Clean date index
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df = df.reset_index().rename(columns={"Date": "date"})

    # Select columns
    keep = ["date", "Open", "High", "Low", "Close", "Volume"]
    df = df[keep].dropna(subset=["Open", "High", "Low", "Close"]).copy()

    # Format
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["Volume"] = df["Volume"].astype("float64")

    # Rename to AV-style headers
    df = df.rename(
        columns={
            "Open": "1. open",
            "High": "2. high",
            "Low": "3. low",
            "Close": "4. close",
            "Volume": "5. volume",
        }
    )

    df = df[["date", "1. open", "2. high", "3. low", "4. close", "5. volume"]]

    # ✅ Reverse to newest → oldest
    df = df[::-1].reset_index(drop=True)

    out_path = os.path.join(OUT_DIR, f"{output_symbol}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {output_symbol} → {out_path}  ({len(df)} rows, newest first)")


if __name__ == "__main__":
    fetch_and_save_yf("HG=F", "COPPER", start=START_DATE, end=END_DATE)
    fetch_and_save_yf("NG=F", "NATURAL_GAS", start=START_DATE, end=END_DATE)
