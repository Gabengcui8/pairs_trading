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

import base64
import html
import os
import pandas as pd
from dataclasses import replace

import pairs_trading as pt

DATA_DIR = "data"
OUT_DIR = "output"
PCT_METRICS = {"Total Return", "CAGR", "Ann. Vol", "Max Drawdown", "Daily Hit Rate"}


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


def period_metrics_frame(period_perfs: dict) -> pd.DataFrame:
    rows = []
    for period, perfs in period_perfs.items():
        for strategy, metrics in perfs.items():
            rows.append({"Period": period.upper(), "Strategy": strategy, **metrics})
    return pd.DataFrame(rows)


def _display_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in PCT_METRICS:
        if col in out:
            out[col] = out[col].map(
                lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "")
    for col in ("Sharpe", "Sortino", "Calmar"):
        if col in out:
            out[col] = out[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    if "Trading Days" in out:
        out["Trading Days"] = out["Trading Days"].map(
            lambda x: f"{int(x)}" if pd.notna(x) else "")
    return out


def _display_comparison(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index().rename(columns={"index": "Metric"})
    for col in out.columns[1:]:
        values = []
        for metric, value in zip(out["Metric"], out[col]):
            if pd.isna(value):
                values.append("")
            elif metric in PCT_METRICS:
                values.append(f"{float(value) * 100:.2f}%")
            elif metric == "Trading Days":
                values.append(f"{int(round(float(value)))}")
            else:
                values.append(f"{float(value):.4f}")
        out[col] = values
    return out


def _image_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{data}"


def write_dashboard(path: str, period_df: pd.DataFrame, comp: pd.DataFrame,
                    selected_pairs: pd.DataFrame, pair_freq: pd.DataFrame):
    """Create a self-contained HTML dashboard for presentation."""
    main = period_df[period_df["Strategy"].eq("optimized_vol_target_garch")]
    risk = period_df[period_df["Strategy"].eq("optimized_risk_balanced_garch")]
    def _metric(strategy, period, metric):
        mask = period_df["Strategy"].eq(strategy) & period_df["Period"].eq(period)
        return float(period_df.loc[mask, metric].iloc[0])

    def _pct(x):
        return f"{x * 100:.2f}%"

    cards = [
        ("Headline Strategy", "optimized_vol_target_garch"),
        ("Pre-COVID Return", _pct(_metric("optimized_vol_target_garch", "PRE", "Total Return"))),
        ("Post-COVID Return", _pct(_metric("optimized_vol_target_garch", "POST", "Total Return"))),
        ("Recent Holdout Return", _pct(_metric("optimized_vol_target_garch", "RECENT", "Total Return"))),
        ("Risk-Balanced Avg Sharpe", f"{risk['Sharpe'].mean():.4f}"),
        ("Dashboard", "self-contained HTML"),
    ]
    card_html = "".join(
        f'<div class="card"><div class="label">{html.escape(k)}</div>'
        f'<div class="value">{html.escape(v)}</div></div>'
        for k, v in cards
    )
    css = """
body{margin:0;background:#f5f7fb;color:#172033;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}
main{max-width:1280px;margin:0 auto;padding:30px 26px 60px}
h1{margin:0 0 6px;font-size:30px}.subtitle{color:#657084;margin-bottom:24px}
.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:18px 0 26px}
.card,section{background:white;border:1px solid #d9e0ea;border-radius:16px;box-shadow:0 4px 14px rgba(31,78,121,.06)}
.card{padding:16px 18px}.label{color:#657084;font-size:13px;margin-bottom:7px}.value{font-size:24px;font-weight:750;color:#1f4e79}
section{padding:20px;margin:18px 0}h2{font-size:20px;margin:0 0 12px}.note{color:#657084;line-height:1.55;margin-top:0}
.table-wrap{overflow:auto;border:1px solid #edf1f6;border-radius:12px}table{width:100%;border-collapse:collapse;font-size:13px}
th,td{border-bottom:1px solid #edf1f6;padding:9px 10px;text-align:right;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{background:#f0f4fa;color:#2b3a55;font-weight:700}
img{max-width:100%;border:1px solid #d9e0ea;border-radius:12px;background:white}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
footer{color:#657084;font-size:12px;margin-top:18px}@media(max-width:900px){.cards,.grid2{grid-template-columns:1fr}}
"""
    html_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pairs Trading Backtest Dashboard</title><style>{css}</style></head><body><main>
<h1>Pairs Trading Strategy Dashboard</h1>
<div class="subtitle">QF621 Group 9 | Latest local backtest output</div>
<div class="cards">{card_html}</div>
<section><h2>Main Strategy Summary</h2>
<p class="note">Headline strategy: Engle-Granger cointegration, same sector/sub-industry filter, VIX gate, GARCH allocation, and 15% volatility target capped at 3x scale.</p>
<div class="table-wrap">{_display_metrics(main).to_html(index=False, escape=False)}</div></section>
<section><h2>Risk-Balanced Companion</h2>
<p class="note">Lower volatility target version for the risk-control story. It sacrifices headline return but improves average Sharpe and reduces drawdown.</p>
<div class="table-wrap">{_display_metrics(risk).to_html(index=False, escape=False)}</div></section>
<section><h2>All Period Metrics</h2><div class="table-wrap">{_display_metrics(period_df).to_html(index=False, escape=False)}</div></section>
<section><h2>Pre-COVID Variant Comparison</h2><div class="table-wrap">{_display_comparison(comp).to_html(index=False, escape=False)}</div></section>
<section><h2>Charts</h2><div class="grid2">
<div><h3>Backtest Overview</h3><img src="{_image_data_uri(os.path.join(OUT_DIR, 'backtest_overview.png'))}" alt="Backtest overview"></div>
<div><h3>Variant Comparison</h3><img src="{_image_data_uri(os.path.join(OUT_DIR, 'variant_comparison.png'))}" alt="Variant comparison"></div>
</div><div style="margin-top:16px"><h3>Pair Selection Heatmap</h3><img src="{_image_data_uri(os.path.join(OUT_DIR, 'pair_selection_heatmap.png'))}" alt="Pair selection heatmap"></div></section>
<section><h2>Selected Pairs By Rolling Window</h2><div class="table-wrap">{selected_pairs.to_html(index=False, escape=False)}</div></section>
<section><h2>Pair Frequency</h2><div class="table-wrap">{pair_freq.to_html(index=False, escape=False)}</div></section>
<footer>No raw data cache is embedded. Chart images are embedded so the dashboard opens as one file.</footer>
</main></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return path


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
        "optimized_risk_balanced_garch": pt.optimized_risk_balanced_config(),
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
        "optimized_risk_balanced_garch": pt.optimized_risk_balanced_config(),
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
                        "optimized_risk_balanced_garch",
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
    period_df = period_metrics_frame(period_perfs)
    period_df.to_csv(os.path.join(OUT_DIR, "period_metrics.csv"), index=False)
    selected_pairs = pd.read_excel(os.path.join(OUT_DIR, "pairs_backtest_results.xlsx"),
                                   sheet_name="Selected Pairs")
    pair_freq = pd.read_excel(os.path.join(OUT_DIR, "pairs_backtest_results.xlsx"),
                              sheet_name="Pair Frequency")
    write_dashboard(os.path.join(OUT_DIR, "backtest_dashboard.html"),
                    period_df, comp, selected_pairs, pair_freq)

    print(f"\nSaved charts, workbook, period metrics, and dashboard to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
