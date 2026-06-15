"""
================================================================================
Pairs Trading Strategy on S&P 500 Stocks  --  Full Backtesting Framework
QF621 Quantitative Trading Strategies, Group 9
================================================================================

A configurable statistical-arbitrage pairs-trading engine that implements EVERY
variant listed in the project proposal (section 6), all driven by `Config`:

  6.1 Asset universe ........ S&P 500 names  OR  liquid ETFs
  6.2 Pair selection ........ cointegration (Engle-Granger) | distance (Gatev
                              2006) | correlation (OLS R^2)
  6.3 Pair restrictions ..... same GICS sector | market-cap band | company age
                              | corporate-finance screen | PCA-cluster matching
  6.4 Trading parameters .... N pairs, entry / exit / stop thresholds,
                              daily vs VIX-adjusted entry (trade more when calm)
  6.5 Capital allocation .... fixed equal | dynamic (concentrate when in cash)
                              | GARCH inverse-volatility

Pipeline (per rolling window):
    build candidates -> apply restrictions -> rank & pick top-N -> z-score
    signals (entry/exit/stop) -> per-pair P&L -> allocate capital -> aggregate.

DATA
----
Prices: yfinance (needs internet). Point-in-time S&P 500 membership, GICS
sectors, market cap and fundamentals should come from CRSP/Compustat via WRDS
(hooks + documented schema below); a static large-cap fallback + real sector
map are provided so the framework runs out of the box.

Requires:  pip install yfinance statsmodels pandas numpy arch scikit-learn
Run:       python pairs_trading.py
================================================================================
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration -- every proposal parameter is a toggle here
# =============================================================================
@dataclass
class Config:
    # --- 6.1 universe & sample ---
    universe: str = "sp500"          # "sp500" | "etf"
    tickers: list = field(default_factory=list)   # override universe if given
    start: str = "2015-01-01"
    end: str = "2019-12-31"

    # --- rolling window (trading days) ---
    formation_days: int = 252        # ~12 months (pair selection)
    trading_days: int = 126          # ~6  months (trading)
    step_days: int = 126             # re-estimate every ~6 months

    # --- 6.2 pair-selection method ---
    method: str = "cointegration"    # "cointegration" | "distance" | "correlation"
    p_value_threshold: float = 0.05  # cointegration significance
    min_r2: float = 0.80             # correlation-method minimum R^2
    n_pairs: int = 10                # top-N pairs
    min_corr: float = 0.50           # cheap pre-filter before the slow coint test
    require_positive_beta: bool = True

    # --- 6.3 pair restrictions (each independently toggleable) ---
    restrict_same_sector: bool = False
    restrict_mcap: bool = False
    mcap_log_tol: float = 0.75       # |ln(mcap_a) - ln(mcap_b)| <= tol
    restrict_age: bool = False
    min_age_years: float = 5.0
    restrict_fundamentals: bool = False
    restrict_pca_cluster: bool = False
    pca_components: int = 5
    pca_clusters: int = 8

    # --- 6.4 trading rules (z-score / std-dev units) ---
    entry_z: float = 2.0
    exit_z: float = 0.0
    stop_z: float = 3.0
    vix_adjust: bool = False         # scale entry threshold by VIX regime
    vix_scale_lo: float = 0.6        # calm  -> lower threshold -> trade more
    vix_scale_hi: float = 1.4        # turbulent -> higher threshold

    # --- 6.5 capital allocation ---
    allocation: str = "equal"        # "equal" | "dynamic" | "garch"

    # --- costs & accounting ---
    tc_bps: float = 10.0             # per leg, per trade (round-trip = 2 legs)
    ann_factor: int = 252


# =============================================================================
# Universes & metadata
#   Sectors below are REAL GICS groupings (reliable). Market cap / fundamentals
#   are illustrative placeholders -- REPLACE with CRSP/Compustat (WRDS) for the
#   real study. PCA clustering and age-from-history need no external data.
# =============================================================================
SP500_FALLBACK = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP",
    "XOM", "CVX", "COP", "SLB",
    "JNJ", "PFE", "MRK", "ABBV", "UNH",
    "PG", "KO", "PEP", "WMT", "COST",
    "HD", "LOW", "MCD", "NKE",
    "T", "VZ", "DIS", "CMCSA",
    "BA", "CAT", "GE", "HON", "MMM",
    "INTC", "CSCO", "ORCL", "IBM", "QCOM", "TXN", "V", "MA",
]

ETF_UNIVERSE = [
    "SPY", "IVV", "VOO", "QQQ", "DIA", "IWM",            # broad index
    "XLF", "XLE", "XLK", "XLV", "XLI", "XLP",            # SPDR sectors
    "XLY", "XLU", "XLB", "XLRE", "XLC",
    "XOP", "OIH", "KRE", "SMH", "SOXX",                 # industry
    "GLD", "SLV", "GDX",                                # metals
    "TLT", "IEF", "SHY", "LQD", "HYG",                  # bonds
    "EEM", "EFA", "VWO", "FXI",                         # international
]

GICS_SECTOR = {
    "AAPL": "InfoTech", "MSFT": "InfoTech", "NVDA": "InfoTech", "INTC": "InfoTech",
    "CSCO": "InfoTech", "ORCL": "InfoTech", "IBM": "InfoTech", "QCOM": "InfoTech",
    "TXN": "InfoTech", "V": "InfoTech", "MA": "InfoTech",
    "GOOGL": "CommServices", "META": "CommServices", "DIS": "CommServices",
    "CMCSA": "CommServices", "T": "CommServices", "VZ": "CommServices",
    "AMZN": "ConsDiscretionary", "HD": "ConsDiscretionary", "LOW": "ConsDiscretionary",
    "MCD": "ConsDiscretionary", "NKE": "ConsDiscretionary",
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "C": "Financials",
    "GS": "Financials", "MS": "Financials", "AXP": "Financials",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "JNJ": "HealthCare", "PFE": "HealthCare", "MRK": "HealthCare",
    "ABBV": "HealthCare", "UNH": "HealthCare",
    "PG": "ConsStaples", "KO": "ConsStaples", "PEP": "ConsStaples",
    "WMT": "ConsStaples", "COST": "ConsStaples",
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
    "HON": "Industrials", "MMM": "Industrials",
}


@dataclass
class Metadata:
    """
    Optional firm characteristics for the 6.3 restrictions. Defaults to the
    static maps above; supply your own (from WRDS) to run the real study.

      sectors      : {ticker -> GICS sector str}
      mcap         : {ticker -> market cap}            (for the mcap band)
      ipo_year     : {ticker -> first-listing year}    (for the age filter)
      fundamentals : DataFrame index=ticker, columns include
                     ['profitability', 'gross_margin', 'cfo']  (corporate screen)
    """
    sectors: dict = field(default_factory=lambda: dict(GICS_SECTOR))
    mcap: dict = field(default_factory=dict)
    ipo_year: dict = field(default_factory=dict)
    fundamentals: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_universe(name: str) -> list:
    """
    Quick-test universes. For the real S&P 500 study, pull point-in-time
    membership from CRSP/WRDS  (crsp_a_indexes.dsp500list joined to
    crsp.stocknames) so each window only sees names alive on those dates.
    """
    return {"sp500": SP500_FALLBACK, "etf": ETF_UNIVERSE}[name]


# =============================================================================
# Data layer
# =============================================================================
def download_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Daily adjusted-close prices via yfinance. Requires internet access."""
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False, group_by="column")
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    px = px.dropna(axis=1, how="all").sort_index()
    keep = px.columns[px.notna().mean() > 0.95]
    return px[keep].ffill().dropna(how="any")


def download_vix(start: str, end: str) -> pd.Series:
    """CBOE VIX (^VIX) close, for the 6.4 VIX-adjusted entry threshold."""
    import yfinance as yf
    raw = yf.download("^VIX", start=start, end=end, auto_adjust=False, progress=False)
    s = raw["Close"]
    return (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s).rename("VIX")


# =============================================================================
# 6.3 Pair restrictions -> candidate generation
# =============================================================================
def _pca_clusters(form_prices: pd.DataFrame, cfg: Config) -> dict:
    """Project names onto top PCA factors of returns, then KMeans-cluster them."""
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    rets = form_prices.pct_change(fill_method=None).dropna()
    if rets.shape[0] < cfg.pca_components + 2 or rets.shape[1] < 2:
        return {c: 0 for c in form_prices.columns}
    X = ((rets - rets.mean()) / rets.std().replace(0, 1)).T.values   # names x time
    k = min(cfg.pca_components, X.shape[1] - 1, X.shape[0] - 1)
    factors = PCA(n_components=max(k, 1)).fit_transform(X)
    n_clusters = min(cfg.pca_clusters, len(form_prices.columns))
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(factors)
    return dict(zip(form_prices.columns, labels))


def build_candidates(form_prices: pd.DataFrame, cfg: Config, meta: Metadata,
                     formation_year: int) -> list:
    """All eligible (a, b) pairs after every enabled 6.3 restriction."""
    # drop degenerate series (constant over the window -> no tradeable spread)
    names = [c for c in form_prices.columns if form_prices[c].std() > 0]

    # company-age screen: enough listing history before the formation window
    if cfg.restrict_age:
        names = [t for t in names
                 if (formation_year - meta.ipo_year.get(t, formation_year)) >= cfg.min_age_years]

    # corporate-finance screen: keep firms passing simple quality thresholds
    if cfg.restrict_fundamentals and not meta.fundamentals.empty:
        f = meta.fundamentals
        ok = []
        for t in names:
            if t not in f.index:
                continue
            row = f.loc[t]
            if (row.get("profitability", 1) > 0 and
                    row.get("gross_margin", 1) > 0.20 and
                    row.get("cfo", 1) > 0):
                ok.append(t)
        names = ok

    clusters = _pca_clusters(form_prices[names], cfg) if cfg.restrict_pca_cluster else None

    out = []
    for a, b in combinations(names, 2):
        if cfg.restrict_same_sector and meta.sectors.get(a) != meta.sectors.get(b):
            continue
        if cfg.restrict_mcap and meta.mcap:
            ma, mb = meta.mcap.get(a), meta.mcap.get(b)
            if ma is None or mb is None or abs(np.log(ma) - np.log(mb)) > cfg.mcap_log_tol:
                continue
        if clusters is not None and clusters.get(a) != clusters.get(b):
            continue
        out.append((a, b))
    return out


# =============================================================================
# 6.2 Pair-selection methods
# =============================================================================
def estimate_hedge_ratio(y: pd.Series, x: pd.Series):
    """Engle-Granger step 1: OLS  y = alpha + beta*x.  Returns (alpha, beta).
    has_constant='add' guarantees an intercept even if x is constant over the
    window (e.g. a halted/floored stock), so we always get two parameters."""
    X = sm.add_constant(np.asarray(x, dtype=float), has_constant="add")
    res = sm.OLS(np.asarray(y, dtype=float), X).fit()
    return float(res.params[0]), float(res.params[1])


def select_pairs(form: pd.DataFrame, candidates: list, cfg: Config) -> list:
    """
    Rank candidate pairs by the chosen method and return the top-N as records.
    Each record carries the info needed to form its trading spread later:
      weight_mode 'beta_shares' (cointegration/correlation): hold 1 share A vs
                  beta shares B; spread = P_A - beta*P_B.
      weight_mode 'dollar_equal' (distance): hold $1 long / $1 short on prices
                  normalised to the formation start; spread = nA - nB.
    """
    recs = []
    for a, b in candidates:
        if cfg.method == "cointegration":
            try:
                _, pval, _ = coint(form[a], form[b])
            except Exception:
                continue
            if pval >= cfg.p_value_threshold:
                continue
            _, beta = estimate_hedge_ratio(form[a], form[b])
            if cfg.require_positive_beta and beta <= 0:
                continue
            recs.append({"a": a, "b": b, "score": pval, "beta": beta,
                         "weight_mode": "beta_shares", "metric": "pvalue"})

        elif cfg.method == "correlation":
            r2 = float(form[a].corr(form[b])) ** 2
            if r2 < cfg.min_r2:
                continue
            _, beta = estimate_hedge_ratio(form[a], form[b])
            if cfg.require_positive_beta and beta <= 0:
                continue
            recs.append({"a": a, "b": b, "score": -r2, "beta": beta,
                         "weight_mode": "beta_shares", "metric": "R2"})

        elif cfg.method == "distance":
            na, nb = form[a] / form[a].iloc[0], form[b] / form[b].iloc[0]
            ssd = float(((na - nb) ** 2).sum())
            recs.append({"a": a, "b": b, "score": ssd,
                         "norm_a": float(form[a].iloc[0]), "norm_b": float(form[b].iloc[0]),
                         "weight_mode": "dollar_equal", "metric": "SSD"})
        else:
            raise ValueError(f"unknown method: {cfg.method}")

    recs.sort(key=lambda r: r["score"])      # lower score = better, for all methods
    return recs[: cfg.n_pairs]


# =============================================================================
# 3. Spread, signals, single-pair backtest
# =============================================================================
def make_spread(prices: pd.DataFrame, rec: dict) -> pd.Series:
    if rec["weight_mode"] == "beta_shares":
        return prices[rec["a"]] - rec["beta"] * prices[rec["b"]]
    return prices[rec["a"]] / rec["norm_a"] - prices[rec["b"]] / rec["norm_b"]


def leg_weights(prices: pd.DataFrame, rec: dict):
    """Prior-day dollar weights of the long/short legs (no look-ahead); gross = 1."""
    if rec["weight_mode"] == "beta_shares":
        pa = prices[rec["a"]].shift(1)
        pb = (rec["beta"] * prices[rec["b"]]).shift(1)
        d = (pa.abs() + pb.abs()).replace(0, np.nan)
        return pa / d, pb / d
    half = pd.Series(0.5, index=prices.index)
    return half, half


def generate_positions(z: pd.Series, cfg: Config, entry=None) -> pd.Series:
    """
    Stateful entry/exit/stop signal, position in {-1, 0, +1}.
      +1 long spread  (long A, short B)  when z <= -entry
      -1 short spread (short A, long B)   when z >= +entry
       0 flat after z reverts through exit_z, or after the stop is breached
    `entry` may be a per-day array (VIX-adjusted) or None (use cfg.entry_z).
    """
    zv = np.asarray(z, dtype=float)
    ev = np.full(len(zv), cfg.entry_z) if entry is None else np.asarray(entry, dtype=float)
    pos = np.zeros(len(zv))
    state = 0
    for t in range(len(zv)):
        if state == 0:
            if zv[t] <= -ev[t]:
                state = 1
            elif zv[t] >= ev[t]:
                state = -1
        elif state == 1:
            if zv[t] >= cfg.exit_z or zv[t] <= -cfg.stop_z:
                state = 0
        elif state == -1:
            if zv[t] <= cfg.exit_z or zv[t] >= cfg.stop_z:
                state = 0
        pos[t] = state
    return pd.Series(pos, index=z.index)


def backtest_pair(seed_slice: pd.DataFrame, rec: dict, mu: float, sd: float,
                  cfg: Config, entry=None) -> pd.DataFrame:
    """
    Per-pair daily results. `seed_slice` = one seed day + the trading window
    (seed seeds pct_change and the first position; caller drops the seed row).
    Returns columns: ret (net of cost), pos, dz (z-score change, for GARCH).
    """
    spread = make_spread(seed_slice, rec)
    z = (spread - mu) / sd
    pos = generate_positions(z, cfg, entry)

    rA = seed_slice[rec["a"]].pct_change(fill_method=None)
    rB = seed_slice[rec["b"]].pct_change(fill_method=None)
    wA, wB = leg_weights(seed_slice, rec)

    gross = pos.shift(1) * (wA * rA - wB * rB)
    tc = cfg.tc_bps / 1e4
    cost = pos.diff().abs() * tc * (wA.abs() + wB.abs())
    return pd.DataFrame({
        "ret": (gross.fillna(0.0) - cost.fillna(0.0)),
        "pos": pos,
        "dz": spread.diff() / sd,
    })


# =============================================================================
# 6.5 Capital allocation
# =============================================================================
def _garch_vol(recs: list, form: pd.DataFrame, dz_df: pd.DataFrame) -> pd.DataFrame:
    """
    GARCH(1,1) one-step-ahead volatility forecast per pair, on the z-scored
    spread change (dimensionless -> comparable across pairs). Fit params on the
    formation window, then roll the conditional-variance recursion forward over
    the trading window using realised dz (info up to t-1 only).
    """
    from arch import arch_model
    cols = {}
    for rec in recs:
        key = f'{rec["a"]}-{rec["b"]}'
        if key not in dz_df:
            continue
        sd = make_spread(form, rec).std()
        dz_form = (make_spread(form, rec).diff().dropna() / sd).values
        dz_form = dz_form - dz_form.mean()
        try:
            fit = arch_model(dz_form, mean="Zero", vol="GARCH", p=1, q=1,
                             rescale=False).fit(disp="off")
            w_ = float(fit.params["omega"])
            a_ = float(fit.params["alpha[1]"])
            b_ = float(fit.params["beta[1]"])
            prev_var = float(np.var(dz_form))
            prev_e2 = float(dz_form[-1] ** 2)
            d = dz_df[key].fillna(0.0).values
            sig = np.empty(len(d))
            for t in range(len(d)):
                v = w_ + a_ * prev_e2 + b_ * prev_var      # forecast for day t
                sig[t] = np.sqrt(max(v, 1e-12))
                prev_var, prev_e2 = v, d[t] ** 2
            cols[key] = pd.Series(sig, index=dz_df.index)
        except Exception:
            cols[key] = dz_df[key].rolling(20, min_periods=5).std().bfill()
    return pd.DataFrame(cols)


def allocate(ret_df: pd.DataFrame, pos_df: pd.DataFrame, dz_df: pd.DataFrame,
             cfg: Config, recs: list, form: pd.DataFrame) -> pd.Series:
    """Combine per-pair returns into a portfolio return under the chosen scheme."""
    if ret_df.empty:
        return pd.Series(dtype=float)

    if cfg.allocation == "equal":                     # fixed 1/N (cash sits idle)
        return ret_df.mul(1.0 / len(recs)).sum(axis=1)

    active = pos_df.abs() > 0
    if cfg.allocation == "dynamic":                   # equal among pairs in a trade
        n = active.sum(axis=1).replace(0, np.nan)
        w = active.div(n, axis=0).fillna(0.0)
        return (ret_df * w).sum(axis=1)

    if cfg.allocation == "garch":                     # inverse-vol among active pairs
        vol = _garch_vol(recs, form, dz_df).reindex_like(ret_df)
        inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
        raw = inv.where(active, 0.0).fillna(0.0)
        s = raw.sum(axis=1).replace(0, np.nan)
        w = raw.div(s, axis=0).fillna(0.0)
        return (ret_df * w).sum(axis=1)

    raise ValueError(f"unknown allocation: {cfg.allocation}")


# =============================================================================
# 4. Rolling-window backtest (orchestration)
# =============================================================================
def _entry_series(seed_idx, form_idx, cfg: Config, vix: pd.Series):
    """Per-day entry threshold over the seed slice (VIX-adjusted if enabled)."""
    if not cfg.vix_adjust or vix is None:
        return None
    ref = vix.reindex(form_idx).median()
    if not np.isfinite(ref) or ref == 0:
        return None
    scale = (vix.reindex(seed_idx) / ref).clip(cfg.vix_scale_lo, cfg.vix_scale_hi)
    return (cfg.entry_z * scale).fillna(cfg.entry_z).values


def run_strategy(prices: pd.DataFrame, cfg: Config,
                 meta: Metadata = None, vix: pd.Series = None) -> dict:
    meta = meta or Metadata()
    idx = prices.index
    n = len(idx)
    port = pd.Series(0.0, index=idx)
    selections = []

    start = 0
    while start + cfg.formation_days + cfg.trading_days <= n:
        f0, f1 = start, start + cfg.formation_days
        t1 = f1 + cfg.trading_days
        form = prices.iloc[f0:f1]
        f_year = idx[f1 - 1].year

        candidates = build_candidates(form, cfg, meta, f_year)
        recs = select_pairs(form, candidates, cfg)
        selections.append({"formation": (idx[f0], idx[f1 - 1]),
                           "trading": (idx[f1], idx[t1 - 1]),
                           "n_candidates": len(candidates),
                           "n_selected": len(recs),
                           "pairs": [(r["a"], r["b"]) for r in recs]})

        if recs:
            seed = prices.iloc[f1 - 1:t1]
            entry = _entry_series(seed.index, form.index, cfg, vix)
            rets, poss, dzs, kept = {}, {}, {}, []
            for rec in recs:
                fs = make_spread(form, rec)
                mu, sd = fs.mean(), fs.std()
                if sd == 0 or np.isnan(sd):
                    continue
                bt = backtest_pair(seed, rec, mu, sd, cfg, entry).iloc[1:]
                key = f'{rec["a"]}-{rec["b"]}'
                rets[key], poss[key], dzs[key] = bt["ret"], bt["pos"], bt["dz"]
                kept.append(rec)
            if kept:
                ret_df = pd.DataFrame(rets); pos_df = pd.DataFrame(poss); dz_df = pd.DataFrame(dzs)
                w = allocate(ret_df, pos_df, dz_df, cfg, kept, form)
                port.loc[w.index] = w.values

        start += cfg.step_days

    return {"returns": port, "selections": selections}


# =============================================================================
# 5. Performance metrics
# =============================================================================
def performance_metrics(rets: pd.Series, cfg: Config) -> dict:
    r = rets.fillna(0.0)
    r = r.loc[r.ne(0).cummax()]
    if r.empty:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / cfg.ann_factor
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std() * np.sqrt(cfg.ann_factor)
    sharpe = (r.mean() * cfg.ann_factor) / vol if vol > 0 else np.nan
    dn = r[r < 0].std() * np.sqrt(cfg.ann_factor)
    sortino = (r.mean() * cfg.ann_factor) / dn if dn > 0 else np.nan
    maxdd = (eq / eq.cummax() - 1).min()
    traded = r[r != 0]
    return {
        "Total Return": float(eq.iloc[-1] - 1), "CAGR": float(cagr),
        "Ann. Vol": float(vol), "Sharpe": float(sharpe), "Sortino": float(sortino),
        "Max Drawdown": float(maxdd),
        "Calmar": float(cagr / abs(maxdd)) if maxdd < 0 else np.nan,
        "Daily Hit Rate": float((traded > 0).mean()) if len(traded) else np.nan,
        "Trading Days": int(len(r)),
    }


# =============================================================================
# 6. Optimisation helpers
# =============================================================================
def parameter_sweep(prices, base, meta=None, vix=None,
                    entry_grid=(1.5, 2.0, 2.5), npairs_grid=(5, 10, 20),
                    stop_grid=(2.5, 3.0, 3.5)) -> pd.DataFrame:
    """6.4 grid-search of the core trading parameters, ranked by Sharpe."""
    rows = []
    for e in entry_grid:
        for npr in npairs_grid:
            for s in stop_grid:
                cfg = replace(base, entry_z=e, n_pairs=npr, stop_z=s)
                p = performance_metrics(run_strategy(prices, cfg, meta, vix)["returns"], cfg)
                if p:
                    rows.append({"entry_z": e, "n_pairs": npr, "stop_z": s,
                                 "Sharpe": p["Sharpe"], "CAGR": p["CAGR"],
                                 "MaxDD": p["Max Drawdown"]})
    return pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)


def run_configs(prices, configs: dict, meta=None, vix=None) -> dict:
    """Run several named Config variants; returns {name: run_strategy result}."""
    return {name: run_strategy(prices, cfg, meta, vix) for name, cfg in configs.items()}


def compare_configs(prices, configs: dict, meta=None, vix=None, results: dict = None):
    """Tabulate metrics for named Config variants side by side (one column each)."""
    results = results or run_configs(prices, configs, meta, vix)
    out = {name: performance_metrics(results[name]["returns"], configs[name]) for name in configs}
    return pd.DataFrame(out).round(4)


# =============================================================================
# 7. Visualisation  (matplotlib; saves PNGs you can drop straight into a report)
# =============================================================================
def _equity_drawdown(returns: pd.Series):
    r = returns.fillna(0.0)
    r = r.loc[r.ne(0).cummax()]
    eq = (1 + r).cumprod()
    return eq, eq / eq.cummax() - 1


def pair_selection_matrix(selections: list) -> pd.DataFrame:
    """Binary matrix (pair x rolling window): was the pair selected that window?"""
    labels = [f"{s['trading'][0].date()}" for s in selections]
    allp = sorted({f"{a}-{b}" for s in selections for a, b in s["pairs"]})
    mat = pd.DataFrame(0, index=allp, columns=labels)
    for lab, s in zip(labels, selections):
        for a, b in s["pairs"]:
            mat.loc[f"{a}-{b}", lab] = 1
    return mat


def plot_backtest(res: dict, title: str = "Pairs Trading Backtest", save_path: str = None):
    """Single-run overview: cumulative NAV, underwater drawdown, pairs/window."""
    import matplotlib.pyplot as plt
    eq, dd = _equity_drawdown(res["returns"])
    sel = res["selections"]
    xs = [s["trading"][0] for s in sel]
    ns = [s["n_selected"] for s in sel]

    fig, ax = plt.subplots(3, 1, figsize=(12, 11),
                           gridspec_kw={"height_ratios": [3, 2, 2]}, sharex=False)
    ax[0].plot(eq.index, eq.values, lw=1.6, color="#1f4e79")
    ax[0].axhline(1, color="grey", lw=0.8, ls="--")
    ax[0].set_ylabel("Growth of $1"); ax[0].set_title(title, fontweight="bold")
    ax[0].grid(alpha=0.3)

    ax[1].fill_between(dd.index, dd.values * 100, 0, color="#c0392b", alpha=0.5)
    ax[1].set_ylabel("Drawdown (%)"); ax[1].grid(alpha=0.3)

    ax[2].bar(xs, ns, width=80, color="#2e7d32", alpha=0.8)
    ax[2].set_ylabel("Pairs selected"); ax[2].set_xlabel("Rolling window (trading start)")
    ax[2].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return save_path


def plot_comparison(results: dict, configs: dict, comp_df: pd.DataFrame, save_path: str = None):
    """Variant comparison: overlaid equity curves + Sharpe / CAGR / MaxDD bars."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, res), c in zip(results.items(), cmap):
        eq, _ = _equity_drawdown(res["returns"])
        ax[0, 0].plot(eq.index, eq.values, lw=1.3, label=name, color=c)
    ax[0, 0].axhline(1, color="grey", lw=0.8, ls="--")
    ax[0, 0].set_title("Equity curves", fontweight="bold")
    ax[0, 0].set_ylabel("Growth of $1"); ax[0, 0].legend(fontsize=7); ax[0, 0].grid(alpha=0.3)

    names = list(comp_df.columns)
    for axi, metric, col in [(ax[0, 1], "Sharpe", "#1f4e79"),
                             (ax[1, 0], "CAGR", "#2e7d32"),
                             (ax[1, 1], "Max Drawdown", "#c0392b")]:
        vals = comp_df.loc[metric].values.astype(float)
        if metric in ("CAGR", "Max Drawdown"):
            vals = vals * 100
        axi.barh(names, vals, color=col, alpha=0.85)
        axi.set_title(metric + (" (%)" if metric != "Sharpe" else ""), fontweight="bold")
        axi.grid(alpha=0.3, axis="x"); axi.invert_yaxis()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return save_path


def plot_pair_heatmap(selections: list, save_path: str = None):
    """Heatmap of which pairs were selected in which window (turnover / stability)."""
    import matplotlib.pyplot as plt
    mat = pair_selection_matrix(selections)
    fig, axi = plt.subplots(figsize=(max(8, 0.5 * mat.shape[1] + 3), max(4, 0.3 * mat.shape[0] + 1)))
    axi.imshow(mat.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axi.set_yticks(range(mat.shape[0])); axi.set_yticklabels(mat.index, fontsize=7)
    axi.set_xticks(range(mat.shape[1])); axi.set_xticklabels(mat.columns, fontsize=7, rotation=90)
    axi.set_title("Pair selection across windows", fontweight="bold")
    axi.set_xlabel("Trading-window start")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return save_path


# =============================================================================
# 8. Excel export  (formatted results workbook -- backtest OUTPUTS, not a model)
# =============================================================================
def export_excel(path: str, comp_df: pd.DataFrame, base_res: dict,
                 sweep_df: pd.DataFrame = None):
    """Write a formatted multi-sheet results workbook."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    HEAD = PatternFill("solid", fgColor="1F4E79")
    HFONT = Font(name="Arial", bold=True, color="FFFFFF", size=11)
    BASE = Font(name="Arial", size=10)
    THIN = Side(style="thin", color="D9D9D9")
    BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
    PCT_ROWS = {"Total Return", "CAGR", "Ann. Vol", "Max Drawdown", "Daily Hit Rate"}

    wb = Workbook()

    # --- Sheet 1: variant comparison (metrics x variants) ---
    ws = wb.active; ws.title = "Variant Comparison"
    ws.cell(1, 1, "Metric").font = HFONT; ws.cell(1, 1).fill = HEAD
    for j, name in enumerate(comp_df.columns, start=2):
        c = ws.cell(1, j, name); c.font = HFONT; c.fill = HEAD; c.alignment = Alignment(horizontal="center")
    for i, metric in enumerate(comp_df.index, start=2):
        ws.cell(i, 1, metric).font = Font(name="Arial", bold=True, size=10)
        for j, name in enumerate(comp_df.columns, start=2):
            v = comp_df.loc[metric, name]
            c = ws.cell(i, j, float(v) if pd.notna(v) else None); c.font = BASE; c.border = BORDER
            c.alignment = Alignment(horizontal="center")
            if metric in PCT_ROWS:
                c.number_format = "0.0%;(0.0%);-"
            elif metric == "Trading Days":
                c.number_format = "0"
            else:
                c.number_format = "0.00;(0.00);-"
    ws.column_dimensions["A"].width = 16
    for j in range(2, comp_df.shape[1] + 2):
        ws.column_dimensions[get_column_letter(j)].width = 16
    ws.freeze_panes = "B2"

    # --- Sheet 2: selected pairs per window (base run) ---
    ws2 = wb.create_sheet("Selected Pairs")
    hdr = ["Formation start", "Formation end", "Trading start", "Trading end",
           "# candidates", "# selected", "Pairs"]
    for j, h in enumerate(hdr, start=1):
        c = ws2.cell(1, j, h); c.font = HFONT; c.fill = HEAD
    for i, s in enumerate(base_res["selections"], start=2):
        ws2.cell(i, 1, str(s["formation"][0].date())); ws2.cell(i, 2, str(s["formation"][1].date()))
        ws2.cell(i, 3, str(s["trading"][0].date())); ws2.cell(i, 4, str(s["trading"][1].date()))
        ws2.cell(i, 5, s["n_candidates"]); ws2.cell(i, 6, s["n_selected"])
        ws2.cell(i, 7, ", ".join(f"{a}-{b}" for a, b in s["pairs"]))
        for j in range(1, 8):
            ws2.cell(i, j).font = BASE
    for j, w in zip(range(1, 8), [15, 15, 14, 14, 12, 11, 70]):
        ws2.column_dimensions[get_column_letter(j)].width = w
    ws2.freeze_panes = "A2"

    # --- Sheet 3: pair selection frequency ---
    ws3 = wb.create_sheet("Pair Frequency")
    freq = pair_selection_matrix(base_res["selections"]).sum(axis=1).sort_values(ascending=False)
    ws3.cell(1, 1, "Pair").font = HFONT; ws3.cell(1, 1).fill = HEAD
    ws3.cell(1, 2, "Windows selected").font = HFONT; ws3.cell(1, 2).fill = HEAD
    for i, (p, n) in enumerate(freq.items(), start=2):
        ws3.cell(i, 1, p).font = BASE; ws3.cell(i, 2, int(n)).font = BASE
    ws3.column_dimensions["A"].width = 16; ws3.column_dimensions["B"].width = 18
    ws3.freeze_panes = "A2"

    # --- Sheet 4: parameter sweep (optional) ---
    if sweep_df is not None and not sweep_df.empty:
        ws4 = wb.create_sheet("Param Sweep")
        for j, h in enumerate(sweep_df.columns, start=1):
            c = ws4.cell(1, j, h); c.font = HFONT; c.fill = HEAD
        for i, (_, row) in enumerate(sweep_df.iterrows(), start=2):
            for j, h in enumerate(sweep_df.columns, start=1):
                c = ws4.cell(i, j, float(row[h])); c.font = BASE
                c.number_format = "0.0%" if h in ("CAGR", "MaxDD") else "0.00" if h == "Sharpe" else "0.0"
        for j in range(1, len(sweep_df.columns) + 1):
            ws4.column_dimensions[get_column_letter(j)].width = 12
        ws4.freeze_panes = "A2"

    wb.save(path)
    return path


# =============================================================================
# Runner
# =============================================================================
def run_period(label, start, end, base, meta=None, vix=None, verbose=True):
    cfg = replace(base, start=start, end=end,
                  tickers=(base.tickers or load_universe(base.universe)))
    prices = download_prices(cfg.tickers, start, end)
    vix_s = vix if vix is not None else (download_vix(start, end) if cfg.vix_adjust else None)
    res = run_strategy(prices, cfg, meta, vix_s)
    perf = performance_metrics(res["returns"], cfg)
    if verbose:
        avg = np.mean([s["n_selected"] for s in res["selections"]]) if res["selections"] else 0
        print(f"\n{'='*62}\n{label}: {start} -> {end}  "
              f"[{cfg.method} | alloc={cfg.allocation} | vix={cfg.vix_adjust}]\n{'='*62}")
        print(f"Universe: {prices.shape[1]} names, {prices.shape[0]} days | "
              f"windows: {len(res['selections'])} | avg pairs/window: {avg:.1f}")
        print("-" * 40)
        for k, v in perf.items():
            print(f"  {k:16s}: {v:,.4f}" if isinstance(v, float) else f"  {k:16s}: {v}")
    return {"prices": prices, "vix": vix_s, **res, "perf": perf}


if __name__ == "__main__":
    base = Config(
        universe="sp500", method="cointegration", allocation="equal",
        formation_days=252, trading_days=126, step_days=126,
        n_pairs=10, entry_z=2.0, exit_z=0.0, stop_z=3.0,
        p_value_threshold=0.05, min_corr=0.5, tc_bps=10.0,
    )
    meta = Metadata()   # static sector map; add mcap / fundamentals / ipo for those filters

    # --- base strategy, both regimes ---
    pre  = run_period("PRE-COVID",  "2015-01-01", "2019-12-31", base, meta)
    post = run_period("POST-COVID", "2022-01-01", "2024-12-31", base, meta)
    print(f"\n{'='*62}\nPRE vs POST-COVID\n{'='*62}")
    print(pd.DataFrame({"Pre-COVID": pre["perf"], "Post-COVID": post["perf"]}).round(4).to_string())

    # --- ablation across the proposal's variants (reuses pre-COVID prices) ---
    px, vx = pre["prices"], pre["vix"]
    variants = {
        "coint/equal":        replace(base),
        "distance/equal":     replace(base, method="distance"),
        "correlation/equal":  replace(base, method="correlation"),
        "coint/same-sector":  replace(base, restrict_same_sector=True),
        "coint/pca-cluster":  replace(base, restrict_pca_cluster=True),
        "coint/dynamic":      replace(base, allocation="dynamic"),
        "coint/garch":        replace(base, allocation="garch"),
        "coint/vix-adjust":   replace(base, vix_adjust=True),
    }
    # VIX needed only for the vix-adjust variant:
    vx = vx if vx is not None else download_vix("2015-01-01", "2019-12-31")
    var_res = run_configs(px, variants, meta, vx)              # run each variant ONCE
    comp = compare_configs(px, variants, meta, vx, results=var_res)
    print(f"\n{'='*62}\nVARIANT COMPARISON (pre-COVID)\n{'='*62}")
    print(comp.to_string())

    # --- charts + results workbook ---
    plot_backtest(pre, "Pairs Trading - Cointegration / Equal (pre-COVID)", "backtest_overview.png")
    plot_comparison(var_res, variants, comp, "variant_comparison.png")
    plot_pair_heatmap(pre["selections"], "pair_selection_heatmap.png")
    # sweep = parameter_sweep(px, base, meta)                  # optional (6.4); slow
    export_excel("pairs_backtest_results.xlsx", comp, pre, sweep_df=None)
    print("\nSaved: backtest_overview.png, variant_comparison.png, "
          "pair_selection_heatmap.png, pairs_backtest_results.xlsx")
