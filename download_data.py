"""
================================================================================
download_data.py  --  fetch & cache every input the backtest needs
QF621 Quantitative Trading Strategies, Group 9
================================================================================

Run ONCE:   python download_data.py
Creates a ./data folder so the backtest can run repeatedly without re-fetching:

    prices_pre.csv,  prices_post.csv     daily adjusted close
                                           (CRSP/WRDS or yfinance fallback)
    vix_pre.csv,     vix_post.csv        ^VIX close (VIX-adjusted entry)
    sectors.csv                          GICS sector per ticker (static map below)
    mcap.csv, ipo.csv, fundamentals.csv  OPTIONAL firm characteristics (WRDS)

The default run uses yfinance + the bundled static sector map, so it works with
no special access. For the REAL study, run with --source wrds. That path pulls
point-in-time S&P 500 membership and CRSP adjusted daily prices from WRDS,
which is the proposal-compliant path for reducing survivorship bias.

Requires:  pip install -r requirements.txt
================================================================================
"""

import argparse
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


def _wrds_connection():
    """Open WRDS connection with a clear install hint."""
    try:
        import wrds
    except ImportError as exc:
        raise RuntimeError(
            "WRDS source requested but the `wrds` package is not installed. "
            "Run `pip install wrds` or `pip install -r requirements.txt`."
        ) from exc
    return wrds.Connection()


def _canonical_tickers_by_permno(constituents: pd.DataFrame) -> dict:
    """
    Stable ticker label per PERMNO.

    CRSP stocknames can contain historical ticker changes. The backtest keys
    membership, metadata, and prices by one column label, so each PERMNO needs a
    canonical ticker. If two PERMNOs map to the same latest ticker, suffix the
    PERMNO to keep columns unique.
    """
    latest = (
        constituents.sort_values(["permno", "end"])
        .drop_duplicates("permno", keep="last")
    )
    base = latest.set_index("permno")["ticker"].to_dict()
    counts = pd.Series(base).value_counts()
    return {
        permno: (ticker if counts.get(ticker, 0) == 1 else f"{ticker}_{int(permno)}")
        for permno, ticker in base.items()
    }


# =============================================================================
# WRDS source  --  Need `wrds` package and a WRDS account.
# =============================================================================
def wrds_sp500_constituents(start, end):
    """
    Point-in-time S&P 500 membership from CRSP (the proposal's survivorship-bias
    fix). Returns permno/ticker with the dates each name was in the index, so
    each formation window can be filtered to names alive on those dates.
    """
    db = _wrds_connection()                     # uses ~/.pgpass or prompts
    q = """
        select distinct a.permno, a.start, a.ending, b.ticker, b.comnam
        from   crsp_a_indexes.dsp500list a
        join   crsp.stocknames b
               on a.permno = b.permno
              and b.namedt <= a.ending
              and b.nameendt >= a.start
        where  a.ending >= %(s)s and a.start <= %(e)s
    """
    df = db.raw_sql(q, params={"s": start, "e": end})
    db.close()
    df = df.rename(columns={"ending": "end"})
    df["ticker"] = df["ticker"].map(normalize_ticker)
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df = df.dropna(subset=["ticker"])
    canonical = _canonical_tickers_by_permno(df)
    df["ticker"] = df["permno"].map(canonical)
    return df.dropna(subset=["ticker"]).drop_duplicates(
        ["permno", "ticker", "start", "end"])


def wrds_company_metadata():
    """GICS sector/sub-industry and IPO year from Compustat company metadata."""
    db = _wrds_connection()
    df = db.raw_sql("""
        select tic as ticker, gsector, gind, gsubind, ipodate
        from comp.company
        where tic is not null
    """)
    db.close()
    df["ticker"] = df["ticker"].map(normalize_ticker)
    df["sector"] = df["gsector"].map(
        lambda x: str(int(x)) if pd.notna(x) else None)
    industry = df["gsubind"].fillna(df["gind"]).fillna(df["gsector"])
    df["industry"] = industry.map(
        lambda x: str(int(x)) if pd.notna(x) else None)
    df["ipo_year"] = pd.to_datetime(df["ipodate"], errors="coerce").dt.year
    return df.drop_duplicates("ticker")


def wrds_crsp_daily_prices(constituents: pd.DataFrame, start, end) -> pd.DataFrame:
    """
    CRSP adjusted daily prices for all historical S&P 500 permnos.

    CRSP stores signed raw prices; adjusted close is abs(prc) / cfacpr. The
    column names use the historical ticker attached to the constituent record.
    When a ticker maps to multiple permnos, duplicate date/ticker observations
    are averaged after adjustment; this keeps the downstream backtest keyed by
    ticker while retaining delisted historical names available in CRSP.
    """
    permnos = sorted({int(p) for p in constituents["permno"].dropna().unique()})
    if not permnos:
        return pd.DataFrame()
    db = _wrds_connection()
    frames = []
    for chunk in _chunks(permnos, 800):
        q = """
            select permno, date, abs(prc) / nullif(cfacpr, 0) as adj_close
            from crsp.dsf
            where date between %(s)s and %(e)s
              and permno in %(permnos)s
              and prc is not null
              and cfacpr is not null
        """
        frames.append(db.raw_sql(q, params={"s": start, "e": end, "permnos": tuple(chunk)}))
    db.close()
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    raw["date"] = pd.to_datetime(raw["date"])
    permno_ticker = (
        constituents.sort_values(["permno", "end"])
        .drop_duplicates("permno", keep="last")
        .set_index("permno")["ticker"]
    )
    raw["ticker"] = raw["permno"].map(permno_ticker).map(normalize_ticker)
    raw = raw.dropna(subset=["ticker", "adj_close"])
    raw = raw.groupby(["date", "ticker"], as_index=False)["adj_close"].mean()
    return raw.pivot(index="date", columns="ticker", values="adj_close").sort_index()


def wrds_market_cap(constituents: pd.DataFrame, asof="2018-12-31"):
    """Latest CRSP market cap snapshot in the 7 calendar days ending at `asof`."""
    permnos = sorted({int(p) for p in constituents["permno"].dropna().unique()})
    if not permnos:
        return pd.DataFrame(columns=["ticker", "mcap_busd"])
    start = (pd.Timestamp(asof) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    db = _wrds_connection()
    q = """
        select permno, date, abs(prc) * shrout / 1e6 as mcap_busd
        from crsp.dsf
        where date between %(s)s and %(e)s
          and permno in %(permnos)s
          and prc is not null and shrout is not null
    """
    df = db.raw_sql(q, params={"s": start, "e": asof, "permnos": tuple(permnos)}).dropna()
    db.close()
    if df.empty:
        return pd.DataFrame(columns=["ticker", "mcap_busd"])
    permno_ticker = (
        constituents.sort_values(["permno", "end"])
        .drop_duplicates("permno", keep="last")
        .set_index("permno")["ticker"]
    )
    latest = df.sort_values("date").drop_duplicates("permno", keep="last")
    latest["ticker"] = latest["permno"].map(permno_ticker).map(normalize_ticker)
    return latest.dropna(subset=["ticker"])[["ticker", "mcap_busd"]]


def wrds_fundamentals(fyear=2018):
    """
    Corporate-finance screen inputs from Compustat annual (comp.funda):
    profitability (ROA), gross margin, and operating cash flow.
    """
    db = _wrds_connection()
    q = """
        select tic as ticker, ni, at, revt, cogs, oancf, fyear
        from   comp.funda
        where  fyear = %(fy)s and indfmt='INDL' and datafmt='STD'
               and popsrc='D' and consol='C'
    """
    f = db.raw_sql(q, params={"fy": fyear})
    db.close()
    f = f.dropna(subset=["at", "revt"])
    f["ticker"] = f["ticker"].map(normalize_ticker)
    f["profitability"] = f["ni"] / f["at"]
    f["gross_margin"] = (f["revt"] - f["cogs"]) / f["revt"]
    f["cfo"] = f["oancf"]
    return f.set_index("ticker")[["profitability", "gross_margin", "cfo"]]


# =============================================================================
# Main: cache everything to ./data
# =============================================================================
def main(source: str = "yfinance", universe: str = UNIVERSE):
    os.makedirs(DATA_DIR, exist_ok=True)
    use_wrds = source == "wrds"
    tickers = ETF_UNIVERSE if universe == "etf" else SP500_FALLBACK

    # ----- universe & sectors -----
    if universe == "sp500" and use_wrds:
        # Proposal-compliant path: point-in-time members + CRSP prices.
        all_start = min(start for start, _ in PERIODS.values())
        all_end = max(end for _, end in PERIODS.values())
        cons = wrds_sp500_constituents(all_start, all_end)
        tickers = sorted(cons["ticker"].dropna().unique().tolist())
        cons[["ticker", "start", "end", "permno", "comnam"]].to_csv(
            os.path.join(DATA_DIR, "membership.csv"), index=False)
        meta = wrds_company_metadata()
        meta.set_index("ticker")["sector"].to_csv(os.path.join(DATA_DIR, "sectors.csv"))
        meta.set_index("ticker")["industry"].to_csv(os.path.join(DATA_DIR, "industries.csv"))
        meta.dropna(subset=["ipo_year"]).set_index("ticker")["ipo_year"].rename("ipo").to_csv(
            os.path.join(DATA_DIR, "ipo.csv"))
        wrds_market_cap(cons).set_index("ticker")["mcap_busd"].rename("mcap").to_csv(
            os.path.join(DATA_DIR, "mcap.csv"))
        wrds_fundamentals().to_csv(os.path.join(DATA_DIR, "fundamentals.csv"))
        print(f"using {len(tickers)} historical S&P 500 symbols from WRDS/CRSP")
    elif universe == "sp500":
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
        print(f"downloading prices  [{label}] {start} -> {end} via {source} ...")
        if use_wrds and universe == "sp500":
            px = wrds_crsp_daily_prices(cons, start, end)
        else:
            px = download_prices(tickers, start, end)
        px.to_csv(os.path.join(DATA_DIR, f"prices_{label}.csv"))
        print(f"  saved {px.shape[1]} names x {px.shape[0]} days")

        print(f"downloading VIX     [{label}] ...")
        vix = download_vix(start, end)
        vix.to_csv(os.path.join(DATA_DIR, f"vix_{label}.csv"))

    print(f"\nDone. Cached files in ./{DATA_DIR}/ -- now run:  python run_backtest.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["yfinance", "wrds"], default="yfinance",
                        help="Use yfinance fallback or WRDS/CRSP data.")
    parser.add_argument("--universe", choices=["sp500", "etf"], default=UNIVERSE)
    args = parser.parse_args()
    main(source=args.source, universe=args.universe)
