"""
examples/make_demo.py  --  regenerate the sample charts + workbook on SYNTHETIC
data (no internet / no WRDS needed). This is only to illustrate the output
format; the numbers are not real. Run from the project root:

    python examples/make_demo.py
"""

import os
import sys

import numpy as np
import pandas as pd
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pairs_trading as pt

OUT = os.path.dirname(os.path.abspath(__file__))


def synthetic_panel(seed=11, T=1100):
    """4-5 truly cointegrated pairs (stationary AR(1) spread) + noise names."""
    np.random.seed(seed)
    dates = pd.bdate_range("2015-01-01", periods=T)

    def rw(s, sig=1.0):
        return s + np.cumsum(np.random.normal(0, sig, T))

    panel, sectors, mcap, ipo = {}, {}, {}, {}
    specs = [(1.3, 50, "Tech"), (0.8, 80, "Tech"), (2.0, 30, "Bank"),
             (1.1, 60, "Bank"), (1.5, 45, "Tech")]
    for i, (beta, base, sec) in enumerate(specs):
        B = rw(base, 0.8)
        eps = np.zeros(T)
        for t in range(1, T):
            eps[t] = 0.85 * eps[t - 1] + np.random.normal(0, 1.5)
        panel[f"A{i}"], panel[f"B{i}"] = beta * B + eps + 10, B
        for nm, m in [(f"A{i}", 80 + 10 * i), (f"B{i}", 75 + 10 * i)]:
            sectors[nm], mcap[nm], ipo[nm] = sec, m, 2000 + i
    for j in range(6):
        nm = f"X{j}"
        panel[nm], sectors[nm], mcap[nm], ipo[nm] = rw(40 + 5 * j, 1.0), "Misc", 200 + j, 2005

    prices = pd.DataFrame(panel, index=dates).clip(lower=1)
    vix = pd.Series(15 + 5 * np.sin(np.linspace(0, 12, T)) + np.random.normal(0, 1, T),
                    index=dates).clip(8, 40)
    meta = pt.Metadata(sectors=sectors, mcap=mcap, ipo_year=ipo)
    return prices, vix, meta


def main():
    prices, vix, meta = synthetic_panel()
    base = pt.Config(method="cointegration", allocation="equal", n_pairs=8,
                     min_corr=0.3, formation_days=252, trading_days=126, step_days=126)
    variants = {
        "coint/equal":       replace(base),
        "distance/equal":    replace(base, method="distance"),
        "correlation/equal": replace(base, method="correlation", min_r2=0.5),
        "same-sector":       replace(base, restrict_same_sector=True),
        "pca-cluster":       replace(base, restrict_pca_cluster=True, pca_clusters=4),
        "dynamic-alloc":     replace(base, allocation="dynamic"),
        "garch-alloc":       replace(base, allocation="garch"),
        "vix-adjust":        replace(base, vix_adjust=True),
    }
    base_res = pt.run_strategy(prices, base, meta, vix)
    var_res = pt.run_configs(prices, variants, meta, vix)
    comp = pt.compare_configs(prices, variants, meta, vix, results=var_res)
    sweep = pt.parameter_sweep(prices, base, meta, vix,
                               entry_grid=(1.5, 2.0, 2.5), npairs_grid=(5, 8),
                               stop_grid=(2.5, 3.0, 3.5))

    pt.plot_backtest(base_res, "Pairs Trading - Cointegration / Equal (DEMO, synthetic data)",
                     os.path.join(OUT, "backtest_overview.png"))
    pt.plot_comparison(var_res, variants, comp, os.path.join(OUT, "variant_comparison.png"))
    pt.plot_pair_heatmap(base_res["selections"], os.path.join(OUT, "pair_selection_heatmap.png"))
    pt.export_excel(os.path.join(OUT, "pairs_backtest_results.xlsx"), comp, base_res, sweep_df=sweep)
    print("Demo outputs written to examples/ (synthetic data).")


if __name__ == "__main__":
    main()
