"""
High-quality visualizations for the NBA Fantasy Value project.

Reads the project's existing CSV outputs (under sim_stats/) and produces a suite
of presentation-ready figures grouped into five themes:

    overview/  - data sanity checks (players per season, retention, PPG dist)
    value/     - fantasy H2H value insights (category breakdown, scarcity, ...)
    thesis/    - real value (Win Shares) vs fantasy value (H2H)
    model/     - projection model evaluation (pred vs actual, errors, CIs)
    draft/     - draft simulation results (standings, team strengths)

Dependencies: matplotlib, numpy, scipy, pandas (no seaborn required).

Usage (run from the repo root):
    py visual_helper/generate_visualizations.py
    py visual_helper/generate_visualizations.py --theme value
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

# Resolve paths relative to the repo root (parent of this file's folder),
# so the script works no matter where it is launched from.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SIM_STATS = REPO_ROOT / "sim_stats"
OUT_ROOT = SCRIPT_DIR / "quality_plots"

THEMES = ["overview", "value", "thesis", "model", "draft"]


# ---------------------------------------------------------------------
# Theme / styling
# ---------------------------------------------------------------------

# Categorical palette. Deliberately colorblind-safe: it avoids using red and
# green to carry meaning together (a key perception rule from the course notes).
# Based on Wong's colorblind-safe palette.
PALETTE = [
    "#0072b2",  # blue
    "#e69f00",  # orange
    "#009e73",  # bluish green
    "#cc79a7",  # reddish purple
    "#56b4e9",  # sky blue
    "#d55e00",  # vermillion
    "#f0e442",  # yellow
    "#999999",  # grey
    "#000000",  # black
]

# Two-direction (diverging) encoding that does NOT rely on red/green: blue for
# one direction, orange for the other, with a light middle (perception rule:
# "use a diverging scheme where light colors represent middle values").
BLUE = "#0072b2"
ORANGE = "#d55e00"
HIGHLIGHT = "#e69f00"

# Perceptually uniform, colorblind-safe colormaps (the course notes recommend
# viridis-family maps over the old non-uniform "jet").
SEQUENTIAL = plt.get_cmap("viridis")
# Diverging map with a light middle; blue<->orange avoids the red/green trap.
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_orange", ["#0072b2", "#f5f5f5", "#d55e00"]
)


def set_theme() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.8,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.labelsize": 12,
        "axes.labelcolor": "#1e293b",
        "axes.labelweight": "medium",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.color": "#0f172a",
        "legend.frameon": False,
        "legend.fontsize": 10,
        "font.family": "DejaVu Sans",
        "figure.titlesize": 17,
        "figure.titleweight": "bold",
    })
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=PALETTE)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save(fig: plt.Figure, theme: str, name: str) -> None:
    out_dir = ensure_dir(OUT_ROOT / theme)
    out_path = out_dir / name
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {theme}/{name}")


def load_csv(name: str) -> pd.DataFrame | None:
    path = SIM_STATS / name
    if not path.exists():
        print(f"  [SKIP] missing file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        print(f"  [SKIP] could not read {name}: {exc}")
        return None


def style_axes(ax: plt.Axes) -> None:
    """Remove top/right spines for a cleaner look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# The 9 H2H category columns as stored in the value CSVs.
H2H_CATS = ["PTS", "REB", "AST", "STL", "BLK", "FG3M", "TOV", "FG", "FT"]


# ---------------------------------------------------------------------
# Theme: overview
# ---------------------------------------------------------------------

def fig_active_players_per_season(season_df: pd.DataFrame) -> None:
    counts = (
        season_df.dropna(subset=["season"])
        .groupby("season")["player_id"]
        .nunique()
        .reset_index(name="n_players")
        .sort_values("season")
    )
    if counts.empty:
        print("  [SKIP] no season data for active players")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    seasons = counts["season"].astype(int).astype(str)
    bars = ax.bar(seasons, counts["n_players"], color=PALETTE[0], width=0.62)
    for rect, val in zip(bars, counts["n_players"]):
        ax.text(rect.get_x() + rect.get_width() / 2, val + max(counts["n_players"]) * 0.01,
                f"{int(val)}", ha="center", va="bottom", fontsize=11, fontweight="bold",
                color="#1e293b")

    ax.set_title("Player counts taper off in earlier seasons, as expected from roster turnover")
    ax.set_xlabel("Season")
    ax.set_ylabel("Number of distinct players")
    ax.set_ylim(0, counts["n_players"].max() * 1.12)
    style_axes(ax)
    fig.text(0.5, -0.02,
             "Distinct players with at least one game log per season (2023-2026), "
             "scraped from Basketball-Reference. A near-constant ~50-player drop per earlier "
             "season is a sanity check on scraping completeness.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "overview", "active_players_per_season.png")


def fig_player_retention(season_df: pd.DataFrame) -> None:
    seasons = sorted(int(s) for s in season_df["season"].dropna().unique())
    if not seasons:
        print("  [SKIP] no seasons for retention")
        return
    latest = seasons[-1]
    latest_players = set(season_df.loc[season_df["season"] == latest, "player_id"])
    if not latest_players:
        print("  [SKIP] no players in latest season")
        return

    rows = []
    for s in seasons:
        players_s = set(season_df.loc[season_df["season"] == s, "player_id"])
        overlap = len(latest_players & players_s)
        rows.append((s, overlap))

    fig, ax = plt.subplots(figsize=(9, 6))
    labels = [str(s) for s, _ in rows]
    vals = [v for _, v in rows]
    colors = [HIGHLIGHT if s == latest else BLUE for s, _ in rows]
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    for rect, val in zip(bars, vals):
        pct = 100.0 * val / len(latest_players)
        ax.text(rect.get_x() + rect.get_width() / 2, val + max(vals) * 0.01,
                f"{val}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color="#1e293b")

    ax.set_title(f"Fewer of the {latest} players appear in each earlier season")
    ax.set_xlabel("Season")
    ax.set_ylabel(f"Players also active in {latest}")
    ax.set_ylim(0, max(vals) * 1.16)
    style_axes(ax)
    fig.text(0.5, -0.02,
             f"Of {len(latest_players)} players active in {latest}, the bars show how many also "
             f"appear in each earlier season.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "overview", "player_retention.png")


def fig_ppg_distribution_by_season(season_df: pd.DataFrame) -> None:
    if "pts_mean" not in season_df.columns:
        print("  [SKIP] no pts_mean column")
        return
    seasons = sorted(int(s) for s in season_df["season"].dropna().unique())
    if not seasons:
        return
    seasons = seasons[:4]  # 2x2 grid

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    for ax in axes:
        ax.set_visible(False)

    for i, s in enumerate(seasons):
        ax = axes[i]
        ax.set_visible(True)
        vals = pd.to_numeric(
            season_df.loc[season_df["season"] == s, "pts_mean"], errors="coerce"
        ).dropna()
        if vals.empty:
            continue
        ax.hist(vals, bins=25, color=PALETTE[0], alpha=0.75, edgecolor="white",
                density=True, label="Players")

        mu, sigma = float(vals.mean()), float(vals.std(ddof=1))
        if sigma > 0:
            xs = np.linspace(vals.min(), vals.max(), 200)
            ax.plot(xs, stats.norm.pdf(xs, mu, sigma), color=ORANGE, lw=2.4,
                    label=f"Normal fit (μ={mu:.1f}, σ={sigma:.1f})")
        ax.axvline(mu, color="#0f172a", ls="--", lw=1.4, label=f"Mean = {mu:.1f}")
        ax.set_title(f"{s} season")
        ax.set_xlabel("Points per game")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8.5)
        style_axes(ax)

    fig.suptitle("Scoring is right-skewed every season: most players sit in single digits", y=0.99)
    fig.text(0.5, -0.01,
             "One panel per season (conditioning on season). Bars are the empirical density of "
             "points per game; the orange curve is a fitted normal for comparison. The right skew "
             "(few high scorers) means a normal is only a rough fit.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "overview", "ppg_distribution_by_season.png")


def run_overview() -> None:
    print("[overview]")
    season_df = load_csv("player_season_stats.csv")
    if season_df is None:
        return
    season_df["season"] = pd.to_numeric(season_df["season"], errors="coerce")
    fig_active_players_per_season(season_df)
    fig_player_retention(season_df)
    fig_ppg_distribution_by_season(season_df)


# ---------------------------------------------------------------------
# Theme: value
# ---------------------------------------------------------------------

def fig_category_breakdown(value_df: pd.DataFrame) -> None:
    cats = [c for c in H2H_CATS if c in value_df.columns]
    if not cats or "rank_H2H" not in value_df.columns:
        print("  [SKIP] missing category/rank columns")
        return

    top = value_df.sort_values("rank_H2H").head(15).copy()
    # A heatmap (position-encoded grid) is used instead of a stacked bar: the
    # course notes warn that stacked bars have a "jiggling baseline" that makes
    # individual segments hard to compare.
    plot = top.set_index("player_id")[cats]

    fig, ax = plt.subplots(figsize=(11, 8.5))
    im = ax.imshow(plot.values, cmap=SEQUENTIAL, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats)
    ax.set_yticks(range(len(plot.index)))
    ax.set_yticklabels(plot.index)
    for i in range(len(plot.index)):
        for j in range(len(cats)):
            v = plot.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v < 0.45 else "#0f172a")
    ax.set_title("Elite fantasy players are well-rounded, yet each leans on different categories")
    ax.set_xlabel("Scoring category")
    ax.set_ylabel("Player (ranked by H2H value, best at top)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Normalized category score (1 = league best, TOV negative = good)")
    ax.grid(False)
    fig.text(0.5, -0.02,
             "Each cell is a player's normalized score in one category (0-1). Brighter = stronger. "
             "Turnovers (TOV) are stored as negative because fewer is better.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "value", "category_breakdown_top15.png")


def fig_category_scarcity(season_df: pd.DataFrame) -> None:
    # Per-game category means -> coefficient of variation across players.
    cat_cols = {
        "PTS": "pts_mean",
        "REB": "trb_mean",
        "AST": "ast_mean",
        "STL": "stl_mean",
        "BLK": "blk_mean",
        "FG3M": "fg3m_mean",
        "TOV": "tov_mean",
    }
    available = {k: v for k, v in cat_cols.items() if v in season_df.columns}
    if not available:
        print("  [SKIP] no per-game mean columns for scarcity")
        return

    rows = []
    for cat, col in available.items():
        vals = pd.to_numeric(season_df[col], errors="coerce").dropna()
        vals = vals[vals >= 0]
        if vals.empty or vals.mean() == 0:
            continue
        cov = vals.std(ddof=1) / vals.mean()
        rows.append((cat, cov, vals.mean()))

    if not rows:
        return
    rows.sort(key=lambda r: r[1])
    cats = [r[0] for r in rows]
    covs = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    norm = (np.array(covs) - min(covs)) / (max(covs) - min(covs) + 1e-9)
    colors = SEQUENTIAL(0.15 + 0.7 * norm)
    bars = ax.bar(cats, covs, color=colors, width=0.66, edgecolor="white")
    for rect, cov in zip(bars, covs):
        ax.text(rect.get_x() + rect.get_width() / 2, cov + max(covs) * 0.01,
                f"{cov:.2f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color="#1e293b")

    ax.set_title("Blocks are by far the scarcest category, so each block swings a matchup more")
    ax.set_xlabel("Statistical category")
    ax.set_ylabel("Coefficient of variation (std / mean)")
    ax.set_ylim(0, max(covs) * 1.14)
    style_axes(ax)
    fig.text(0.5, -0.03,
             "Higher = scarcer / more spread out across players. Scarce, high-variance categories "
             "(e.g. blocks) swing weekly H2H matchups more than common, evenly-spread ones.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "value", "category_scarcity.png")


def fig_category_correlation(value_df: pd.DataFrame) -> None:
    cats = [c for c in H2H_CATS if c in value_df.columns]
    if len(cats) < 2:
        print("  [SKIP] not enough category columns for correlation")
        return
    data = value_df[cats].apply(pd.to_numeric, errors="coerce")
    corr = data.corr().values

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(cats)))
    ax.set_yticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_yticklabels(cats)
    for i in range(len(cats)):
        for j in range(len(cats)):
            v = corr[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.6 else "#0f172a", fontsize=9)
    ax.set_title("Most fantasy categories are only weakly related, so each must be targeted")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    ax.grid(False)
    fig.text(0.5, -0.02,
             "Pearson correlation between players' normalized category scores. Light = uncorrelated. "
             "Few strong pairs means a balanced roster needs players drafted for specific categories.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "value", "category_correlation_heatmap.png")


def fig_top_h2h_value(value_df: pd.DataFrame) -> None:
    if "H2H_value" not in value_df.columns:
        print("  [SKIP] no H2H_value column")
        return
    top = value_df.sort_values("H2H_value", ascending=False).head(30).copy()
    top = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 11))
    norm = (top["H2H_value"] - top["H2H_value"].min()) / (
        top["H2H_value"].max() - top["H2H_value"].min() + 1e-9)
    colors = SEQUENTIAL(0.2 + 0.7 * norm.values)
    ax.barh(top["player_id"], top["H2H_value"], color=colors, edgecolor="white",
            height=0.72)
    for y, val in enumerate(top["H2H_value"]):
        ax.text(val + top["H2H_value"].max() * 0.005, y, f"{val:.2f}",
                va="center", fontsize=8.5, color="#1e293b")

    ax.set_title("A thin elite tier tops the projected H2H value rankings")
    ax.set_xlabel("Projected H2H value")
    ax.set_ylabel("Player")
    ax.set_xlim(0, top["H2H_value"].max() * 1.08)
    style_axes(ax)
    save(fig, "value", "top30_h2h_value.png")


def run_value() -> None:
    print("[value]")
    value_df = load_csv("h2h_value_2027.csv")
    season_df = load_csv("player_season_stats.csv")
    if value_df is not None:
        fig_category_breakdown(value_df)
        fig_category_correlation(value_df)
        fig_top_h2h_value(value_df)
    if season_df is not None:
        fig_category_scarcity(season_df)


# ---------------------------------------------------------------------
# Theme: thesis (real value vs fantasy value)
# ---------------------------------------------------------------------

def fig_ws_vs_h2h_scatter(ws_df: pd.DataFrame) -> None:
    needed = {"win_shares", "H2H_value"}
    if not needed.issubset(ws_df.columns):
        print("  [SKIP] missing win_shares/H2H_value")
        return
    df = ws_df.dropna(subset=["win_shares", "H2H_value"]).copy()
    x = pd.to_numeric(df["win_shares"], errors="coerce")
    y = pd.to_numeric(df["H2H_value"], errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask].values, y[mask].values
    if len(x) < 3:
        return

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.scatter(x, y, s=42, color=BLUE, alpha=0.6, edgecolor="white", linewidth=0.6,
               zorder=3)

    slope, intercept, r, _, _ = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color=ORANGE, lw=2.6, zorder=4,
            label=f"Trend (Pearson r = {r:.2f})")

    # Annotate a few notable players (largest divergence from the trend).
    if "player_id" in df.columns:
        resid = y - (slope * x + intercept)
        order = np.argsort(np.abs(resid))[::-1][:6]
        ids = df.loc[mask, "player_id"].values
        for idx in order:
            ax.annotate(ids[idx], (x[idx], y[idx]), fontsize=8.5, color="#475569",
                        xytext=(5, 4), textcoords="offset points")

    ax.set_title(f"Fantasy value only moderately tracks real-life value (r = {r:.2f})")
    ax.set_xlabel("Win Shares (real-life basketball value)")
    ax.set_ylabel("H2H value (fantasy value)")
    ax.legend()
    style_axes(ax)
    fig.text(0.5, -0.02,
             "Each point is a player. Points far from the trend line are valued very differently "
             "in fantasy than in real life.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "thesis", "ws_vs_h2h_scatter.png")


def fig_over_under_valued(ws_df: pd.DataFrame) -> None:
    if "rank_diff" not in ws_df.columns or "player_id" not in ws_df.columns:
        print("  [SKIP] missing rank_diff/player_id")
        return
    df = ws_df.dropna(subset=["rank_diff"]).copy()
    df["rank_diff"] = pd.to_numeric(df["rank_diff"], errors="coerce")
    df = df.dropna(subset=["rank_diff"])
    if df.empty:
        return

    top_over = df.sort_values("rank_diff", ascending=False).head(12)
    top_under = df.sort_values("rank_diff").head(12)
    combined = pd.concat([top_under, top_over]).drop_duplicates("player_id")
    combined = combined.sort_values("rank_diff")

    fig, ax = plt.subplots(figsize=(11, 9))
    # Blue / orange instead of green / red so the chart stays colorblind-safe.
    colors = [BLUE if v > 0 else ORANGE for v in combined["rank_diff"]]
    ax.barh(combined["player_id"], combined["rank_diff"], color=colors,
            edgecolor="white", height=0.72)
    ax.axvline(0, color="#0f172a", lw=1.0)
    for y, val in enumerate(combined["rank_diff"]):
        offset = 1 if val >= 0 else -1
        ax.text(val + offset, y, f"{val:+.0f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=8.5, color="#1e293b")

    ax.set_title("Many players are valued very differently in fantasy than in real life")
    ax.set_xlabel("Rank difference  (Win Shares rank − H2H rank)")
    ax.set_ylabel("Player")
    style_axes(ax)

    # Manual legend.
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=BLUE, label="Fantasy value > real value (fantasy overvalued)"),
        Patch(color=ORANGE, label="Real value > fantasy value (fantasy undervalued)"),
    ], loc="lower right")
    fig.text(0.5, -0.02,
             "Players with the largest gap between their real-life rank (Win Shares) and their "
             "fantasy rank (H2H). Large gaps highlight draft bargains and traps.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "thesis", "over_under_valued.png")


def run_thesis() -> None:
    print("[thesis]")
    ws_df = load_csv("ws_vs_h2h_2027.csv")
    if ws_df is None:
        return
    fig_ws_vs_h2h_scatter(ws_df)
    fig_over_under_valued(ws_df)


# ---------------------------------------------------------------------
# Theme: model evaluation
# ---------------------------------------------------------------------

MODEL_STATS = [
    ("PTS", "pred_PTS_mean", "actual_PTS_mean"),
    ("REB", "pred_REB_mean", "actual_REB_mean"),
    ("AST", "pred_AST_mean", "actual_AST_mean"),
    ("STL", "pred_STL_mean", "actual_STL_mean"),
    ("BLK", "pred_BLK_mean", "actual_BLK_mean"),
    ("TOV", "pred_TOV_mean", "actual_TOV_mean"),
    ("FG3M", "pred_FG3M_mean", "actual_FG3M_mean"),
    ("FG%", "pred_FG%_mean", "actual_FG%_mean"),
    ("FT%", "pred_FT%_mean", "actual_FT%_mean"),
]


def fig_pred_vs_actual(eval_df: pd.DataFrame) -> None:
    usable = [(lbl, p, a) for lbl, p, a in MODEL_STATS
              if p in eval_df.columns and a in eval_df.columns]
    if not usable:
        print("  [SKIP] no pred/actual columns")
        return

    n = len(usable)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes:
        ax.set_visible(False)

    for i, (lbl, pcol, acol) in enumerate(usable):
        ax = axes[i]
        ax.set_visible(True)
        pred = pd.to_numeric(eval_df[pcol], errors="coerce")
        act = pd.to_numeric(eval_df[acol], errors="coerce")
        m = pred.notna() & act.notna()
        pred, act = pred[m].values, act[m].values
        if len(pred) < 2:
            continue
        ax.scatter(act, pred, s=24, color=BLUE, alpha=0.5, edgecolor="white",
                   linewidth=0.4, zorder=3)
        lo = float(min(act.min(), pred.min()))
        hi = float(max(act.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], color="#0f172a", ls="--", lw=1.3, zorder=4,
                label="perfect (y = x)")
        r = np.corrcoef(act, pred)[0, 1]
        ax.set_title(f"{lbl}  (r = {r:.2f})", fontsize=12)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend(fontsize=8)
        style_axes(ax)

    fig.suptitle("Projections track actual production closely for volume stats, less so for rates",
                 y=0.997)
    fig.text(0.5, -0.01,
             "One panel per projected statistic (2026). Each point is a player; the dashed line is "
             "a perfect prediction (y = x). Tight clustering around the line means accurate "
             "projections.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "model", "predicted_vs_actual.png")


def fig_error_metrics(summary_df: pd.DataFrame) -> None:
    if "stat" not in summary_df.columns:
        print("  [SKIP] no stat column in summary")
        return
    df = summary_df.copy()

    # MAE / RMSE grouped bar (counting raw stats only; pct stats are tiny scale,
    # but we keep all and let the chart speak).
    if {"mae", "rmse"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(12, 6.5))
        x = np.arange(len(df))
        w = 0.4
        ax.bar(x - w / 2, df["mae"], w, label="MAE", color=BLUE, edgecolor="white")
        ax.bar(x + w / 2, df["rmse"], w, label="RMSE", color=ORANGE, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(df["stat"], rotation=30, ha="right")
        ax.set_title("Projection error is largest for points and tiny for rate stats")
        ax.set_ylabel("Error (in each stat's own units)")
        ax.legend()
        style_axes(ax)
        fig.text(0.5, -0.04,
                 "MAE = mean absolute error, RMSE = root mean squared error (per stat, in that "
                 "stat's units). RMSE above MAE indicates occasional large misses.",
                 ha="center", fontsize=9, color="#64748b")
        save(fig, "model", "error_mae_rmse.png")

    # Pearson / Spearman grouped bar.
    if {"pearson", "spearman"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(12, 6.5))
        x = np.arange(len(df))
        w = 0.4
        ax.bar(x - w / 2, df["pearson"], w, label="Pearson", color=BLUE,
               edgecolor="white")
        ax.bar(x + w / 2, df["spearman"], w, label="Spearman", color=ORANGE,
               edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(df["stat"], rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title("Projections correlate strongly with reality for counting stats, weakly for percentages")
        ax.set_ylabel("Correlation with actual (0-1)")
        ax.legend()
        style_axes(ax)
        fig.text(0.5, -0.04,
                 "Pearson (linear) and Spearman (rank) correlation between projected and actual "
                 "values per stat. Higher is better; percentage stats (FG%, FT%) are hardest to "
                 "predict.",
                 ha="center", fontsize=9, color="#64748b")
        save(fig, "model", "correlation_pearson_spearman.png")


def fig_ci_calibration(summary_df: pd.DataFrame) -> None:
    if not {"stat", "ci_5_95_coverage"}.issubset(summary_df.columns):
        print("  [SKIP] no ci coverage column")
        return
    df = summary_df.copy()
    cov = pd.to_numeric(df["ci_5_95_coverage"], errors="coerce")

    fig, ax = plt.subplots(figsize=(12, 6.5))
    bars = ax.bar(df["stat"], cov, color=BLUE, edgecolor="white", width=0.66)
    ax.axhline(0.90, color=ORANGE, ls="--", lw=2.0, label="ideal coverage = 0.90")
    for rect, c in zip(bars, cov):
        ax.text(rect.get_x() + rect.get_width() / 2, c + 0.01, f"{c:.2f}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                color="#1e293b")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["stat"], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Confidence intervals are too narrow: most stats fall short of 90% coverage")
    ax.set_ylabel("Share of actuals inside predicted 5-95% CI")
    ax.legend()
    style_axes(ax)
    fig.text(0.5, -0.03,
             "Bars below the dashed line mean the model's stated uncertainty is too tight "
             "(real outcomes land outside the interval more than 10% of the time).",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "model", "ci_calibration.png")


def run_model() -> None:
    print("[model]")
    eval_df = load_csv("evaluation_2026.csv")
    summary_df = load_csv("evaluation_2026_summary.csv")
    if eval_df is not None:
        fig_pred_vs_actual(eval_df)
    if summary_df is not None:
        fig_error_metrics(summary_df)
        fig_ci_calibration(summary_df)


# ---------------------------------------------------------------------
# Theme: draft
# ---------------------------------------------------------------------

def fig_standings(stand_df: pd.DataFrame) -> None:
    if not {"team", "W"}.issubset(stand_df.columns):
        print("  [SKIP] missing team/W in standings")
        return
    df = stand_df.sort_values("W", ascending=True).copy()
    labels = [f"Team {int(t)}" if pd.notna(t) and str(t).replace('.', '').isdigit()
              else str(t) for t in df["team"]]

    fig, ax = plt.subplots(figsize=(10, 7.5))
    norm = (df["W"] - df["W"].min()) / (df["W"].max() - df["W"].min() + 1e-9)
    colors = SEQUENTIAL(0.2 + 0.7 * norm.values)
    bars = ax.barh(labels, df["W"], color=colors, edgecolor="white", height=0.7)
    has_cat = "cat_pts" in df.columns
    for y, (rect, (_, row)) in enumerate(zip(bars, df.iterrows())):
        txt = f"{row['W']:.1f} W"
        if has_cat:
            txt += f"  |  {row['cat_pts']:.1f} cat pts"
        ax.text(row["W"] + df["W"].max() * 0.01, y, txt, va="center", fontsize=9,
                color="#1e293b")

    ax.set_title("Win totals separate the simulated league into clear tiers")
    ax.set_xlabel("Average weekly wins over the simulated season")
    ax.set_ylabel("Team (draft strategy)")
    ax.set_xlim(0, df["W"].max() * 1.18)
    style_axes(ax)
    fig.text(0.5, -0.02,
             "Each team drafted with a different strategy; bars show average weekly category wins "
             "(out of 9), with total category points to the right.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "draft", "standings_wins.png")


def fig_team_strengths(avg_df: pd.DataFrame, stand_df: pd.DataFrame | None) -> None:
    cat_cols = [c for c in ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M",
                            "FG_pct", "FT_pct"] if c in avg_df.columns]
    if "team" not in avg_df.columns or len(cat_cols) < 2:
        print("  [SKIP] missing team/category columns for strengths")
        return

    df = avg_df.copy()
    # Order teams by standings (wins) if available, best at top.
    if stand_df is not None and {"team", "W"}.issubset(stand_df.columns):
        order = stand_df.sort_values("W", ascending=False)["team"].tolist()
        df["__ord"] = df["team"].apply(lambda t: order.index(t) if t in order else 1e9)
        df = df.sort_values("__ord")

    data = df[cat_cols].apply(pd.to_numeric, errors="coerce")
    # Column-wise z-score so each category is on the same relative scale.
    # TOV is bad: flip its sign so "strong" is always warm/high.
    z = (data - data.mean()) / (data.std(ddof=0) + 1e-9)
    if "TOV" in z.columns:
        z["TOV"] = -z["TOV"]

    team_labels = [f"Team {int(t)}" if pd.notna(t) and str(t).replace('.', '').isdigit()
                   else str(t) for t in df["team"]]

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(z.values, cmap=DIVERGING, vmin=-2.2, vmax=2.2, aspect="auto")
    ax.set_xticks(range(len(cat_cols)))
    display_cats = [c.replace("_pct", "%") for c in cat_cols]
    ax.set_xticklabels(display_cats, rotation=30, ha="right")
    ax.set_yticks(range(len(team_labels)))
    ax.set_yticklabels(team_labels)
    for i in range(len(team_labels)):
        for j in range(len(cat_cols)):
            v = z.values[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    color="white" if abs(v) > 1.3 else "#1e293b", fontsize=8)
    ax.set_title("Each simulated team is built around a different set of category strengths")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Relative strength (standard deviations from league mean)")
    ax.grid(False)
    fig.text(0.5, -0.02,
             "Teams (rows, ordered best-to-worst by standings) vs the nine categories. Cells are "
             "z-scores within each category; turnovers are sign-flipped so warmer/positive always "
             "means stronger.",
             ha="center", fontsize=9, color="#64748b")
    save(fig, "draft", "team_category_strengths.png")


def run_draft() -> None:
    print("[draft]")
    stand_df = load_csv("draft_standings_2027.csv")
    avg_df = load_csv("draft_team_averages_2027.csv")
    if stand_df is not None:
        fig_standings(stand_df)
    if avg_df is not None:
        fig_team_strengths(avg_df, stand_df)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

RUNNERS = {
    "overview": run_overview,
    "value": run_value,
    "thesis": run_thesis,
    "model": run_model,
    "draft": run_draft,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--theme", choices=THEMES, default=None,
                        help="Only generate one theme (default: all).")
    args = parser.parse_args()

    set_theme()
    ensure_dir(OUT_ROOT)

    themes = [args.theme] if args.theme else THEMES
    print(f"Output directory: {OUT_ROOT}")
    for theme in themes:
        RUNNERS[theme]()

    print("\nDone. Figures saved under:", OUT_ROOT)


if __name__ == "__main__":
    main()
