"""
================================================================================
run_backtest.py  --  load cached data, run the full study, save charts + Excel
QF621 Quantitative Trading Strategies, Group 9
================================================================================

Run AFTER download_data.py:   python run_backtest.py

Loads ./data/*, runs the base strategy on both regimes, runs the 6.2/6.3/6.5
variant ablation, and writes to ./output/:
    backtest_overview.png        equity / drawdown / pairs-per-window
    variant_comparison.png       equity overlay + Sharpe / CAGR / MaxDD bars
    pair_selection_heatmap.png   which pairs were selected in which window
    pairs_backtest_results.xlsx  formatted results workbook (4 sheets)
================================================================================
"""

import os
import pandas as pd
from dataclasses import replace

import pairs_trading as pt

DATA_DIR = "data"
OUT_DIR = "output"


# =============================================================================
# Cache loaders
# =============================================================================
def load_prices(label) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, f"prices_{label}.csv"),
                       index_col=0, parse_dates=True)


def load_vix(label):
    path = os.path.join(DATA_DIR, f"vix_{label}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]


def load_metadata() -> pt.Metadata:
    def _read_map(fname, col=0):
        p = os.path.join(DATA_DIR, fname)
        return pd.read_csv(p, index_col=0).iloc[:, col].to_dict() if os.path.exists(p) else {}

    sectors = _read_map("sectors.csv") or dict(pt.GICS_SECTOR)
    industries = _read_map("industries.csv")
    mcap = _read_map("mcap.csv")
    funda_path = os.path.join(DATA_DIR, "fundamentals.csv")
    funda = pd.read_csv(funda_path, index_col=0) if os.path.exists(funda_path) else pd.DataFrame()
    ipo = _read_map("ipo.csv")
    membership_path = os.path.join(DATA_DIR, "membership.csv")
    membership = (pd.read_csv(membership_path, parse_dates=["start", "end"])
                  if os.path.exists(membership_path) else pd.DataFrame())
    return pt.Metadata(sectors=sectors, industries=industries, mcap=mcap, ipo_year=ipo,
                       fundamentals=funda, membership=membership)


# =============================================================================
# Main
# =============================================================================
def main():
    if not os.path.exists(os.path.join(DATA_DIR, "prices_pre.csv")):
        print("No cached data found. Run  python download_data.py  first.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)

    meta = load_metadata()
    configs = {
        "proposal_baseline": pt.proposal_baseline_config(),
        "optimized_equal_1x": pt.optimized_proposal_config(),
        "optimized_aggressive_3x": pt.optimized_aggressive_config(),
        "optimized_vol_target_garch": pt.optimized_vol_target_config(),
        "optimized_vol_target_garch_20bps": pt.optimized_vol_target_config(tc_bps=20.0),
    }

    # ----- strict proposal baseline + validated optimized variants -----
    period_results = {}
    period_perfs = {}
    optimized_pre_res = None
    labels = [label for label in ("pre", "post", "recent")
              if os.path.exists(os.path.join(DATA_DIR, f"prices_{label}.csv"))]
    for label in labels:
        px = load_prices(label)
        vix = load_vix(label)
        period_results[label] = pt.run_configs(px, configs, meta, vix)
        period_perfs[label] = {
            name: pt.performance_metrics(result["returns"], configs[name])
            for name, result in period_results[label].items()
        }
        if label == "pre":
            optimized_pre_res, pre_px, pre_vix = (
                period_results[label]["optimized_vol_target_garch"], px, vix)

        print(f"\n=== {label.upper()} RESULTS ===")
        print(pd.DataFrame(period_perfs[label]).round(4).to_string())

    print("\n=== PERIOD COMPARISON ===")
    for name in configs:
        table = pd.DataFrame({
            label.upper(): period_perfs[label][name] for label in labels
        })
        print(f"\n{name}")
        print(table.round(4).to_string())

    # ----- variant ablation on pre-COVID prices (sections 6.2 / 6.3 / 6.5 / 6.4) -----
    base = pt.optimized_proposal_config()
    variants = {
        "optimized_equal_1x": base,
        "optimized_aggressive_3x": pt.optimized_aggressive_config(),
        "optimized_vol_target_garch": pt.optimized_vol_target_config(),
        "optimized_vol_target_garch_20bps": pt.optimized_vol_target_config(tc_bps=20.0),
        "proposal_baseline": configs["proposal_baseline"],
        "proposal_n5_dynamic": pt.optimized_proposal_config(n_pairs=5, allocation="dynamic"),
        "distance_same_industry": replace(base, method="distance"),
        "correlation_log_ratio": replace(
            base,
            method="correlation",
            correlation_on_returns=True,
            correlation_spread_mode="log_ratio",
            min_r2=0.25,
            min_recent_corr=0.30,
            require_spread_reversion=True,
        ),
    }
    var_res = {
        "optimized_vol_target_garch": optimized_pre_res,
        **{name: period_results["pre"][name]
           for name in ("optimized_equal_1x", "optimized_aggressive_3x",
                        "optimized_vol_target_garch_20bps",
                        "proposal_baseline")},
    }
    missing = {k: v for k, v in variants.items() if k not in var_res}
    var_res.update(pt.run_configs(pre_px, missing, meta, pre_vix))
    comp = pt.compare_configs(pre_px, variants, meta, pre_vix, results=var_res)
    print("\n=== VARIANT COMPARISON (pre-COVID) ===")
    print(comp.to_string())

    # ----- optional parameter sweep (section 6.4) -----
    # sweep = pt.parameter_sweep(pre_px, base, meta, pre_vix)
    sweep = None

    # ----- charts + workbook -----
    pt.plot_backtest(optimized_pre_res,
                     "Pairs Trading - Optimized GARCH Vol-Target Cointegration (pre-COVID)",
                     os.path.join(OUT_DIR, "backtest_overview.png"))
    pt.plot_comparison(var_res, variants, comp,
                       os.path.join(OUT_DIR, "variant_comparison.png"))
    pt.plot_pair_heatmap(optimized_pre_res["selections"],
                         os.path.join(OUT_DIR, "pair_selection_heatmap.png"))
    pt.export_excel(os.path.join(OUT_DIR, "pairs_backtest_results.xlsx"),
                    comp, optimized_pre_res, sweep_df=sweep)

    print(f"\nSaved charts + workbook to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
