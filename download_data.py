"""
================================================================================
download_data.py  --  fetch & cache every input the backtest needs
QF621 Quantitative Trading Strategies, Group 9
================================================================================

Run ONCE:   python download_data.py
Creates a ./data folder so the backtest can run repeatedly without re-fetching:

    prices_pre.csv,  prices_post.csv     daily adjusted-close (yfinance)
    vix_pre.csv,     vix_post.csv        ^VIX close (for VIX-adjusted entry)
    sectors.csv                          GICS sector per ticker (static map below)
    mcap.csv, ipo.csv, fundamentals.csv  OPTIONAL firm characteristics (WRDS)

The default run uses yfinance + the bundled static sector map, so it works with
no special access. For the REAL study, fill in the WRDS functions further down
(point-in-time S&P 500 membership + Compustat fundamentals) using your WRDS
login -- these avoid survivorship bias and power the 6.3 restriction filters.

Requires:  pip install yfinance pandas    (+ `wrds` only if you use the hooks)
================================================================================
"""

import os
from itertools import islice
from io import StringIO
import pandas as pd

from pairs_trading import SP500_FALLBACK, ETF_UNIVERSE, GICS_SECTOR

DATA_DIR = "data"

# Backtest periods from the proposal (section 5)
PERIODS = {
    "pre":  ("2015-01-01", "2019-12-31"),   # pre-COVID
    "post": ("2022-01-01", "2024-12-31"),   # post-COVID
    "recent": ("2024-01-01", "2026-06-16"), # formation in 2024, holdout after
}

# Choose "sp500" (default) or "etf" (6.1 ETF-based pairs)
UNIVERSE = "sp500"


# =============================================================================
# yfinance downloads (no special access required)
# =============================================================================
def normalize_ticker(ticker):
    """Map data-vendor ticker punctuation to Yahoo Finance notation."""
    return str(ticker).strip().upper().replace(".", "-")


def current_sp500_constituents():
    """
    Current S&P 500 members and GICS sectors.

    This is only a no-WRDS fallback and therefore has survivorship bias for
    historical tests. The WRDS path below is the proposal-compliant source.
    """
    import requests
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 pairs-trading-research/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    table = pd.read_html(StringIO(response.text),
                         attrs={"id": "constituents"})[0]
    out = table[["Symbol", "GICS Sector", "GICS Sub-Industry", "Date added"]].copy()
    out.columns = ["ticker", "sector", "industry", "date_added"]
    out["ticker"] = out["ticker"].map(normalize_ticker)
    return out.drop_duplicates("ticker").sort_values("ticker")


def _chunks(values, size):
    it = iter(values)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


def download_prices(tickers, start, end, chunk_size=75) -> pd.DataFrame:
    """Daily adjusted closes, downloaded in chunks and kept as a sparse panel."""
    import yfinance as yf
    frames = []
    tickers = [normalize_ticker(t) for t in tickers]
    for chunk in _chunks(tickers, chunk_size):
        raw = yf.download(chunk, start=start, end=end, auto_adjust=True,
                          progress=False, group_by="column", threads=True)
        if raw.empty:
            continue
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        if isinstance(close, pd.Series):
            close = close.to_frame(name=chunk[0])
        frames.append(close)
    if not frames:
        return pd.DataFrame()
    px = pd.concat(frames, axis=1)
    px = px.loc[:, ~px.columns.duplicated()].sort_index()
    px.columns = [normalize_ticker(c) for c in px.columns]
    return px.dropna(axis=1, how="all").ffill(limit=5)


def download_vix(start, end) -> pd.Series:
    """CBOE VIX (^VIX) close, used by the 6.4 VIX-adjusted entry threshold."""
    import yfinance as yf
    raw = yf.download("^VIX", start=start, end=end, auto_adjust=False, progress=False)
    s = raw["Close"]
    return (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s).rename("VIX")


# =============================================================================
# WRDS hooks  --  OPTIONAL.  Need `pip install wrds` and a WRDS account.
# These return tidy DataFrames; wire their output into the CSVs in main().
# Each query is a sensible starting point -- align fiscal dates to your
# formation windows for the real study.
# =============================================================================
def wrds_sp500_constituents(start, end):
    """
    Point-in-time S&P 500 membership from CRSP (the proposal's survivorship-bias
    fix). Returns permno/ticker with the dates each name was in the index, so
    each formation window can be filtered to names alive on those dates.
    """
    import wrds
    db = wrds.Connection()                      # uses ~/.pgpass or prompts
    q = """
        select a.permno, a.start, a.ending, b.ticker, b.comnam
        from   crsp_a_indexes.dsp500list a
        join   crsp.stocknames b on a.permno = b.permno
        where  a.ending >= %(s)s and a.start <= %(e)s
    """
    df = db.raw_sql(q, params={"s": start, "e": end})
    db.close()
    df = df.rename(columns={"ending": "end"})
    df["ticker"] = df["ticker"].map(normalize_ticker)
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df


def wrds_sectors():
    """GICS sector code per ticker from Compustat (comp.company)."""
    import wrds
    db = wrds.Connection()
    df = db.raw_sql("select tic as ticker, gsector from comp.company where tic is not null")
    db.close()
    df["gsector"] = df["gsector"].astype("Int64")
    return df


def wrds_market_cap(asof="2018-12-31"):
    """Market cap (|prc| * shrout) snapshot from CRSP daily file (crsp.dsf)."""
    import wrds
    db = wrds.Connection()
    q = """
        select b.ticker, abs(a.prc) * a.shrout / 1e6 as mcap_busd
        from   crsp.dsf a join crsp.stocknames b on a.permno = b.permno
        where  a.date = %(d)s
    """
    df = db.raw_sql(q, params={"d": asof}).dropna()
    db.close()
    return df


def wrds_fundamentals(fyear=2018):
    """
    Corporate-finance screen inputs from Compustat annual (comp.funda):
    profitability (ROA), gross margin, and operating cash flow.
    """
    import wrds
    db = wrds.Connection()
    q = """
        select tic as ticker, ni, at, revt, cogs, oancf, fyear
        from   comp.funda
        where  fyear = %(fy)s and indfmt='INDL' and datafmt='STD'
               and popsrc='D' and consol='C'
    """
    f = db.raw_sql(q, params={"fy": fyear})
    db.close()
    f = f.dropna(subset=["at", "revt"])
    f["profitability"] = f["ni"] / f["at"]
    f["gross_margin"] = (f["revt"] - f["cogs"]) / f["revt"]
    f["cfo"] = f["oancf"]
    return f.set_index("ticker")[["profitability", "gross_margin", "cfo"]]


# =============================================================================
# Main: cache everything to ./data
# =============================================================================
def main(use_wrds: bool = False):
    os.makedirs(DATA_DIR, exist_ok=True)
    tickers = ETF_UNIVERSE if UNIVERSE == "etf" else SP500_FALLBACK

    # ----- universe & sectors -----
    if UNIVERSE == "sp500" and use_wrds:
        # Replace the fallback list with point-in-time constituents.
        all_start = min(start for start, _ in PERIODS.values())
        all_end = max(end for _, end in PERIODS.values())
        cons = wrds_sp500_constituents(all_start, all_end)
        tickers = sorted(cons["ticker"].dropna().unique().tolist())
        cons[["ticker", "start", "end", "permno", "comnam"]].to_csv(
            os.path.join(DATA_DIR, "membership.csv"), index=False)
        sec = wrds_sectors().dropna()
        sec["ticker"] = sec["ticker"].map(normalize_ticker)
        pd.Series(sec.set_index("ticker")["gsector"].to_dict(), name="sector").to_csv(
            os.path.join(DATA_DIR, "sectors.csv"))
        wrds_market_cap().set_index("ticker")["mcap_busd"].rename("mcap").to_csv(
            os.path.join(DATA_DIR, "mcap.csv"))
        wrds_fundamentals().to_csv(os.path.join(DATA_DIR, "fundamentals.csv"))
    elif UNIVERSE == "sp500":
        try:
            current = current_sp500_constituents()
            tickers = current["ticker"].tolist()
            current.to_csv(os.path.join(DATA_DIR, "universe_current.csv"), index=False)
            current.set_index("ticker")["sector"].to_csv(
                os.path.join(DATA_DIR, "sectors.csv"))
            current.set_index("ticker")["industry"].to_csv(
                os.path.join(DATA_DIR, "industries.csv"))
            fallback_membership = current[["ticker", "date_added"]].copy()
            fallback_membership["start"] = pd.to_datetime(
                fallback_membership["date_added"], errors="coerce"
            ).fillna(pd.Timestamp("1900-01-01"))
            fallback_membership["end"] = pd.NaT
            fallback_membership[["ticker", "start", "end"]].to_csv(
                os.path.join(DATA_DIR, "membership.csv"), index=False)
            print(f"using {len(tickers)} current S&P 500 symbols "
                  "(survivorship-biased fallback; use WRDS for final study)")
        except Exception as exc:
            print(f"current S&P 500 download failed ({exc}); using bundled fallback")
            pd.Series(GICS_SECTOR, name="sector").to_csv(
                os.path.join(DATA_DIR, "sectors.csv"))
    else:
        pd.Series(GICS_SECTOR, name="sector").to_csv(os.path.join(DATA_DIR, "sectors.csv"))

    # ----- prices & VIX for each period -----
    for label, (start, end) in PERIODS.items():
        print(f"downloading prices  [{label}] {start} -> {end} ...")
        px = download_prices(tickers, start, end)
        px.to_csv(os.path.join(DATA_DIR, f"prices_{label}.csv"))
        print(f"  saved {px.shape[1]} names x {px.shape[0]} days")

        print(f"downloading VIX     [{label}] ...")
        vix = download_vix(start, end)
        vix.to_csv(os.path.join(DATA_DIR, f"vix_{label}.csv"))

    print(f"\nDone. Cached files in ./{DATA_DIR}/ -- now run:  python run_backtest.py")


if __name__ == "__main__":
    # Set use_wrds=True after filling in your WRDS credentials for the real study.
    main(use_wrds=False)
