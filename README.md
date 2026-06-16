# Pairs Trading Strategy on S&P 500 Stocks

QF621 Quantitative Trading Strategies - Group 9

This repository implements the project proposal as a reproducible pairs-trading
backtest:

- Full S&P 500 universe fallback from Wikipedia/yfinance, with WRDS hooks for
  point-in-time CRSP constituents and Compustat/CRSP firm characteristics.
- Engle-Granger two-step cointegration as the strict baseline.
- 12-month formation window, 6-month trading window, rolling every 6 months.
- Entry/exit/stop z-score rules, transaction costs, VIX gating, GARCH risk
  allocation, and volatility targeting.
- Distance, correlation, and cointegration pair-selection variants for the
  section 6 optimization comparison.

## Key Result

The strict proposal baseline was not economically viable after costs. The
validated optimized versions keep the proposal's core cointegration logic but
add cost-aware robustness:

- log-price Engle-Granger cointegration
- same GICS sector and same GICS sub-industry
- positive hedge ratio
- p-value below 0.01
- top 3 pairs
- entry 2.0, exit 0.0, stop 3.5
- 60-trading-day max holding period
- VIX entry block when current VIX exceeds 1.5x the formation median
- GARCH inverse-volatility allocation with a 15% annualized portfolio
  volatility target, capped at 3x scale
- 10 bps transaction cost per leg

Three optimized books are reported:

- `optimized_equal_1x`: conservative 50% long / 50% short style gross exposure.
- `optimized_aggressive_3x`: higher-deployment market-neutral book, about
  150% long and 150% short gross exposure. This keeps the same signals and
  pair selection; it only deploys more gross capital to the market-neutral book.
- `optimized_vol_target_garch`: the main report version. It keeps the same
  pair-selection and trade rules, then allocates active pairs by GARCH
  forecast risk and scales the book to a trailing volatility target.

Recent run from cached data:

| Period | Strict Baseline Return | Optimized 1x Return | Aggressive 3x Return | GARCH Vol-Target Return | GARCH Vol-Target Max DD |
|---|---:|---:|---:|---:|---:|
| 2015-2019 pre-COVID | -35.93% | 4.51% | 13.17% | 20.56% | -14.16% |
| 2022-2024 post-COVID | -12.95% | 2.31% | 6.57% | 13.12% | -11.05% |
| 2025-2026 holdout | -11.45% | 4.23% | 12.99% | 30.52% | -7.56% |

Cost stress: `optimized_vol_target_garch_20bps` remains positive in all three
periods: +2.29%, +5.66%, and +24.87%. The cost-stress drawdown is meaningfully
wider in the pre-COVID period, so this is reported as a robustness check rather
than the headline trading-cost assumption.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

Use cached data if it already exists:

```bash
python run_backtest.py
```

Download fresh yfinance/Wikipedia fallback data, then run:

```bash
python download_data.py
python run_backtest.py
```

The runner writes:

| File | Contents |
|---|---|
| `output/backtest_overview.png` | Equity curve, drawdown, pairs per rolling window |
| `output/variant_comparison.png` | Variant equity and metric comparison |
| `output/pair_selection_heatmap.png` | Selected pairs by window |
| `output/pairs_backtest_results.xlsx` | Formatted result workbook |

Offline engine check:

```bash
python tests/test_synthetic.py
```

## Tencent Cloud Quick Start

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip

git clone https://github.com/Gabengcui8/pairs_trading.git
cd pairs_trading

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python download_data.py
python run_backtest.py
```

If the server has no GUI, download the generated files from `output/` with
`scp`, for example:

```bash
scp -r ubuntu@YOUR_SERVER_IP:~/pairs_trading/output ./pairs_output
```

## Proposal Mapping

| Proposal section | Implementation |
|---|---|
| 6.1 Asset universe | `universe="sp500"` or `"etf"` |
| 6.2 Pair methodology | `method="cointegration"`, `"distance"`, or `"correlation"` |
| 6.3 Restrictions | `restrict_same_sector`, `restrict_same_industry`, `restrict_mcap`, `restrict_age`, `restrict_fundamentals`, `restrict_pca_cluster` |
| 6.4 Trading parameters | `n_pairs`, `entry_z`, `exit_z`, `stop_z`, `vix_adjust`, `max_vix_ratio` |
| 6.5 Capital allocation | `allocation="equal"`, `"dynamic"`, or `"garch"`; optional `vol_target_ann` |
| Realism | `tc_bps`, borrow/financing costs, no look-ahead target accounting |

## Real Data Notes

The default no-WRDS path uses current S&P 500 members from Wikipedia and
yfinance adjusted closes. This is convenient for replication but still has
survivorship bias. For a final institutional-quality study, enable the WRDS
hooks in `download_data.py` to use:

- CRSP `crsp_a_indexes.dsp500list` for point-in-time S&P 500 membership
- CRSP/Compustat for market cap and fundamentals
- Compustat company metadata for GICS classifications

The code already filters each formation window through `membership.csv` when a
point-in-time membership file is available.
