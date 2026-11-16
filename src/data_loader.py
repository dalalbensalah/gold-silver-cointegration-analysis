import pandas as pd
import yfinance as yf


# Tickers utilisés dans l'étude (cf. notebooks 01 et 03)
ASSETS_BIVARIE = {
    "gold": "GC=F",
    "silver": "SI=F",
}

ASSETS_MULTIVARIE = {
    "gold": "GC=F",
    "silver": "SI=F",
    "dxy": "DX-Y.NYB",
    "us10y": "^TNX",
}


def download_asset(
    ticker: str,
    start: str,
    end: str,
    column: str = "Close",
) -> pd.Series:
    """Download a price series via yfinance."""

    data = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )

    if data.empty:
        raise ValueError(f"No data found for {ticker}")

    prices = data[column]

    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]

    prices.name = ticker

    return prices


def load_market_data(
    assets: dict[str, str],
    start: str = "2000-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Downloads market data for multiple assets.

    Parameters
    ----------
    assets : dict
        Dictionary {name: Yahoo Finance ticker}.
    start : str
        Start date.
    end : str
        End date.

    Returns
    -------
    pd.DataFrame
        DataFrame containing raw (non-imputed) price series,
        sorted by date, one column per asset. Imputation and
        transformations (log, returns...) are handled in
        preprocessing.py.
    """

    data = {}

    for name, ticker in assets.items():
        data[name] = download_asset(ticker, start, end)

    df = pd.DataFrame(data)
    df = df.sort_index()

    return df

def save_raw(df: pd.DataFrame, filename: str, output_dir: str = "../data/raw") -> None:
    """Save raw data"""
    path = f"{output_dir}/{filename}"
    df.to_csv(path)
    print(f"Raw data saved : {path}")