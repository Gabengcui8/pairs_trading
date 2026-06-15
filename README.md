# Pairs Trading Strategy on S&P 500 Stocks

**QF621 Quantitative Trading Strategies — Group 9**

A configurable statistical-arbitrage pairs-trading backtester. Cointegrated
pairs are identified on a 12-month formation window, traded on the following
6-month window via a z-score spread signal, and the window is rolled every
6 months. Every variant in the project proposal (section 6) is a `Config`
toggle, so each can be tested by changing one line.

---

## Setup

```bash
pip install -r requirements.txt
```

## Run (two steps)

```bash
python download_data.py     # 1) fetch & cache prices + VIX into ./data/
python run_backtest.py      # 2) backtest + charts + Excel into ./output/
```

`run_backtest.py` produces:

| File | Contents |
|------|----------|
| `output/backtest_overview.png` | Cumulative NAV, drawdown, pairs selected per window |
| `output/variant_comparison.png` | Equity overlay + Sharpe / CAGR / Max-Drawdown bars |
| `output/pair_selection_heatmap.png` | Which pairs were chosen in which window |
| `output/pairs_backtest_results.xlsx` | Results workbook: variant comparison, selected pairs, pair frequency, parameter sweep |

> `examples/` holds the same four outputs generated on **synthetic data** so you
> can see the format before downloading the real data.

A one-shot alternative that downloads inline (no caching) is also available:
```bash
python pairs_trading.py
```

---

## Files

```
pairs_trading.py     core framework (library): data, pair selection, signals,
                     backtest, allocation, metrics, plotting, Excel export
download_data.py     fetch & cache yfinance prices + VIX; WRDS query hooks
run_backtest.py      load cache -> run study -> save charts + workbook
tests/               offline engine validation on synthetic data (no internet)
examples/            sample charts + workbook (synthetic data)
```

Verify the engine offline (no network needed):
```bash
python tests/test_synthetic.py
```

---

## What each proposal section maps to (all in `Config`)

| Proposal | `Config` field(s) | Options |
|----------|-------------------|---------|
| 6.1 Universe | `universe` | `"sp500"` / `"etf"` |
| 6.2 Pair selection | `method` | `"cointegration"` (Engle-Granger) / `"distance"` (Gatev SSD) / `"correlation"` (OLS R²) |
| 6.3 Restrictions | `restrict_same_sector`, `restrict_mcap`, `restrict_age`, `restrict_fundamentals`, `restrict_pca_cluster` | each `True`/`False` + thresholds |
| 6.4 Trading params | `n_pairs`, `entry_z`, `exit_z`, `stop_z`, `vix_adjust` | numeric / bool |
| 6.5 Allocation | `allocation` | `"equal"` / `"dynamic"` / `"garch"` |

Example — switch to the distance method with a same-sector restriction and
GARCH allocation:
```python
from dataclasses import replace
import pairs_trading as pt
cfg = pt.Config(method="distance", restrict_same_sector=True, allocation="garch")
```

---

## Using real data (WRDS)

The default run uses yfinance prices + a bundled static GICS sector map, so it
runs with no special access. For the real study (and to avoid survivorship
bias), enable the WRDS hooks in `download_data.py`:

1. `pip install wrds`, then set `use_wrds=True` at the bottom of `download_data.py`.
2. The provided queries pull point-in-time S&P 500 membership
   (`crsp_a_indexes.dsp500list`), GICS sectors and market cap (CRSP/Compustat),
   and fundamentals (`comp.funda`: ROA, gross margin, operating cash flow) for
   the 6.3 market-cap / age / fundamentals filters.

---

## Method notes (for the write-up)

- **Hedge ratio & P&L.** Cointegration/correlation pairs use the OLS share
  hedge (1 share A vs β shares B); legs are converted to prior-day dollar
  weights so the return reflects the actual hedge with no look-ahead. The
  distance method trades a $1-long/$1-short spread on prices normalised to the
  formation start. Spreads are z-scored with formation-window mean/σ.
- **VIX-adjusted entry (6.4).** The entry threshold is scaled by
  `VIX_t / median(VIX over formation)`, clipped to `[0.6, 1.4]` — calmer
  regimes lower the threshold and trade more often.
- **GARCH allocation (6.5).** A GARCH(1,1) is fit to each pair's standardised
  spread change (dimensionless, comparable across pairs) on the formation
  window; conditional variance is rolled forward over the trading window to
  forecast next-day volatility, and active pairs are weighted ∝ 1/σ.
- **Costs.** 10 bps per leg per trade, charged on every position change.
