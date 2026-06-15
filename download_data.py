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
import pandas as pd

from pairs_trading import SP500_FALLBACK, ETF_UNIVERSE, GICS_SECTOR

DATA_DIR = "data"

# Backtest periods from the proposal (section 5)
PERIODS = {
    "pre":  ("2015-01-01", "2019-12-31"),   # pre-COVID
    "post": ("2022-01-01", "2024-12-31"),   # post-COVID
}

# Choose "sp500" (default) or "etf" (6.1 ETF-based pairs)
UNIVERSE = "sp500"


# =============================================================================
# yfinance downloads (no special access required)
# =============================================================================
def download_prices(tickers, start, end) -> pd.DataFrame:
    """Daily adjusted-close prices; drops names with sparse history."""
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False, group_by="column")
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    px = px.dropna(axis=1, how="all").sort_index()
    keep = px.columns[px.notna().mean() > 0.95]
    return px[keep].ffill().dropna(how="any")


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
    tickers = {"sp500": SP500_FALLBACK, "etf": ETF_UNIVERSE}[UNIVERSE]

    # ----- universe & sectors -----
    if use_wrds:
        # Replace the fallback list with point-in-time constituents.
        cons = wrds_sp500_constituents(*PERIODS["pre"])
        tickers = sorted(cons["ticker"].dropna().unique().tolist())
        sec = wrds_sectors().dropna()
        pd.Series(sec.set_index("ticker")["gsector"].to_dict(), name="sector").to_csv(
            os.path.join(DATA_DIR, "sectors.csv"))
        wrds_market_cap().set_index("ticker")["mcap_busd"].rename("mcap").to_csv(
            os.path.join(DATA_DIR, "mcap.csv"))
        wrds_fundamentals().to_csv(os.path.join(DATA_DIR, "fundamentals.csv"))
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
