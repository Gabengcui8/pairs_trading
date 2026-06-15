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
    mcap = _read_map("mcap.csv")
    funda_path = os.path.join(DATA_DIR, "fundamentals.csv")
    funda = pd.read_csv(funda_path, index_col=0) if os.path.exists(funda_path) else pd.DataFrame()
    ipo = _read_map("ipo.csv")
    return pt.Metadata(sectors=sectors, mcap=mcap, ipo_year=ipo, fundamentals=funda)


# =============================================================================
# Main
# =============================================================================
def main():
    if not os.path.exists(os.path.join(DATA_DIR, "prices_pre.csv")):
        print("No cached data found. Run  python download_data.py  first.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)

    base = pt.Config(
        method="correlation", allocation="dynamic",
        formation_days=252, trading_days=126, step_days=126,
        n_pairs=5, entry_z=2.25, exit_z=0.25, stop_z=3.5,
        p_value_threshold=0.05, min_corr=0.5, min_r2=0.80, tc_bps=10.0,
        restrict_same_sector=True,
        recent_p_value_threshold=0.10,
        min_half_life=2.0, max_half_life=60.0, min_mean_crossings=4,
        reentry_z=1.0, max_holding_days=60,
    )
    meta = load_metadata()

    # ----- base strategy on both regimes -----
    perfs = {}
    base_pre_res = None
    for label in ("pre", "post"):
        px = load_prices(label)
        vix = load_vix(label)
        res = pt.run_strategy(px, base, meta, vix)
        perf = pt.performance_metrics(res["returns"], base)
        perfs[label] = perf
        if label == "pre":
            base_pre_res, pre_px, pre_vix = res, px, vix
        print(f"\n=== {label.upper()}-COVID ({base.method}/{base.allocation}) ===")
        for k, v in perf.items():
            print(f"  {k:16s}: {v:,.4f}" if isinstance(v, float) else f"  {k:16s}: {v}")

    print("\n=== PRE vs POST-COVID ===")
    print(pd.DataFrame({"Pre-COVID": perfs["pre"], "Post-COVID": perfs["post"]}).round(4).to_string())

    # ----- variant ablation on pre-COVID prices (sections 6.2 / 6.3 / 6.5 / 6.4) -----
    variants = {
        "correlation/dynamic": replace(base),
        "correlation/equal": replace(base, allocation="equal"),
        "cointegration/equal": replace(base, method="cointegration", allocation="equal"),
        "distance/equal":    replace(base, method="distance", allocation="equal"),
        "all-sectors":       replace(base, restrict_same_sector=False),
        "pca-cluster":       replace(base, restrict_pca_cluster=True),
        "garch-alloc":       replace(base, allocation="garch"),
        "vix-adjust":        replace(base, vix_adjust=True),
    }
    var_res = pt.run_configs(pre_px, variants, meta, pre_vix)
    comp = pt.compare_configs(pre_px, variants, meta, pre_vix, results=var_res)
    print("\n=== VARIANT COMPARISON (pre-COVID) ===")
    print(comp.to_string())

    # ----- optional parameter sweep (section 6.4) -----
    # sweep = pt.parameter_sweep(pre_px, base, meta, pre_vix)
    sweep = None

    # ----- charts + workbook -----
    pt.plot_backtest(base_pre_res, "Pairs Trading - Correlation / Dynamic (pre-COVID)",
                     os.path.join(OUT_DIR, "backtest_overview.png"))
    pt.plot_comparison(var_res, variants, comp,
                       os.path.join(OUT_DIR, "variant_comparison.png"))
    pt.plot_pair_heatmap(base_pre_res["selections"],
                         os.path.join(OUT_DIR, "pair_selection_heatmap.png"))
    pt.export_excel(os.path.join(OUT_DIR, "pairs_backtest_results.xlsx"),
                    comp, base_pre_res, sweep_df=sweep)

    print(f"\nSaved charts + workbook to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
