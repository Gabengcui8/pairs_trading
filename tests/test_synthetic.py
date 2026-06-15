"""
tests/test_synthetic.py  --  validate every proposal-6 feature OFFLINE on
synthetic data (no internet, no WRDS). Run from the project root:

    python tests/test_synthetic.py
"""

import os
import sys

import numpy as np
import pandas as pd
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pairs_trading as pt


def build_data(seed=7, T=1100):
    np.random.seed(seed)
    dates = pd.bdate_range("2015-01-01", periods=T)

    def rw(s, sig=1.0):
        return s + np.cumsum(np.random.normal(0, sig, T))

    panel, sectors, mcap, ipo = {}, {}, {}, {}
    specs = [(1.3, 50, "Tech"), (0.8, 80, "Tech"), (2.0, 30, "Bank"), (1.1, 60, "Bank")]
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
        panel[nm], sectors[nm], mcap[nm], ipo[nm] = rw(40 + 5 * j, 1.0), "Misc", 200 + j, 2013
    prices = pd.DataFrame(panel, index=dates).clip(lower=1)
    funda = pd.DataFrame({"profitability": 1, "gross_margin": 0.4, "cfo": 1}, index=list(panel))
    meta = pt.Metadata(sectors=sectors, mcap=mcap, ipo_year=ipo, fundamentals=funda)
    vix = pd.Series(15 + 5 * np.sin(np.linspace(0, 12, T)) + np.random.normal(0, 1, T),
                    index=dates).clip(8, 40)
    return prices, meta, vix


def main():
    prices, meta, vix = build_data()
    form = prices.iloc[:252]
    fy = prices.index[251].year
    base = pt.Config(n_pairs=10, min_corr=0.3, formation_days=252,
                     trading_days=126, step_days=126,
                     recent_p_value_threshold=0.20,
                     min_half_life=0.5, max_half_life=200,
                     min_mean_crossings=1, use_log_prices=False)

    # 6.3 restrictions reduce the candidate set
    n_none = len(pt.build_candidates(form, base, meta, fy))
    n_sec = len(pt.build_candidates(form, replace(base, restrict_same_sector=True), meta, fy))
    n_age = len(pt.build_candidates(form, replace(base, restrict_age=True, min_age_years=5), meta, fy))
    assert n_sec < n_none and n_age < n_none, "restrictions did not filter"

    # 6.2 every method recovers true cointegrated pairs in its top set
    true = {tuple(sorted([f"A{i}", f"B{i}"])) for i in range(4)}
    for m in ("cointegration", "distance", "correlation"):
        recs = pt.select_pairs(form, pt.build_candidates(form, replace(base, method=m), meta, fy),
                               replace(base, method=m, min_r2=0.5))
        got = {tuple(sorted([r["a"], r["b"]])) for r in recs}
        assert len(true & got) >= 3, f"{m}: recovered too few true pairs"

    # 6.4 VIX-adjusted entry produces a per-day threshold that changes trades
    z = pd.Series(np.linspace(-3, 3, 60), index=prices.index[:60])
    ent = (base.entry_z * (vix.iloc[:60] / vix.iloc[:252].median()).clip(0.6, 1.4)).values
    assert (pt.generate_positions(z, base).values != pt.generate_positions(z, base, ent).values).any()

    # A stopped spread must normalise before it can enter again.
    z_stop = pd.Series([0, -2.1, -3.2, -3.1, -2.8, -0.8, -2.2])
    p_stop = pt.generate_positions(z_stop, base)
    assert p_stop.tolist() == [0, 1, 0, 0, 0, 0, 1], p_stop.tolist()

    # Formation diagnostics distinguish reversion from one-way drift.
    reverting = pd.Series(np.sin(np.linspace(0, 20, 252)))
    drifting = pd.Series(np.arange(252, dtype=float))
    assert np.isfinite(pt.spread_diagnostics(reverting)["half_life"])
    assert pt.spread_diagnostics(drifting)["half_life"] > 1e6

    # 6.5 every allocation runs end-to-end and trades
    variants = {
        "coint/equal":  replace(base),
        "distance":     replace(base, method="distance"),
        "correlation":  replace(base, method="correlation", min_r2=0.5),
        "same-sector":  replace(base, restrict_same_sector=True),
        "pca":          replace(base, restrict_pca_cluster=True, pca_clusters=4),
        "dynamic":      replace(base, allocation="dynamic"),
        "garch":        replace(base, allocation="garch"),
        "vix":          replace(base, vix_adjust=True),
    }
    for name, cfg in variants.items():
        r = pt.run_strategy(prices, cfg, meta, vix)["returns"]
        assert (r != 0).sum() > 0, f"{name}: produced no trades"
        assert pt.performance_metrics(r, cfg), f"{name}: no metrics"

    print(f"candidates: none={n_none}, same-sector={n_sec}, age>=5y={n_age}")
    print("all 6.1-6.5 features validated on synthetic data  OK")


if __name__ == "__main__":
    main()
