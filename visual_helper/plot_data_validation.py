import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

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
BLUE = "#0072b2"
ORANGE = "#d55e00"
HIGHLIGHT = "#e69f00"

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
        "axes.titlesize": 20,
        "axes.titleweight": "bold",
        "axes.titlepad": 16,
        "axes.labelsize": 18,
        "axes.labelcolor": "#1e293b",
        "axes.labelweight": "medium",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "text.color": "#0f172a",
        "legend.frameon": False,
        "legend.fontsize": 14,
        "font.family": "DejaVu Sans",
        "figure.titlesize": 22,
        "figure.titleweight": "bold",
    })

def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def fig_active_players_per_season(season_df: pd.DataFrame, out_path: Path) -> None:
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

    fig, ax = plt.subplots(figsize=(10, 7))
    seasons = counts["season"].astype(int).astype(str)
    bars = ax.bar(seasons, counts["n_players"], color=PALETTE[0], width=0.62)
    for rect, val in zip(bars, counts["n_players"]):
        ax.text(rect.get_x() + rect.get_width() / 2, val + max(counts["n_players"]) * 0.01,
                f"{int(val)}", ha="center", va="bottom", fontsize=15, fontweight="bold",
                color="#1e293b")

    ax.set_title("Active Players Per Season", fontsize=20, fontweight="bold")
    ax.set_xlabel("Season", fontsize=18)
    ax.set_ylabel("Number of distinct players", fontsize=18)
    ax.set_ylim(0, counts["n_players"].max() * 1.12)
    style_axes(ax)
    fig.text(0.5, -0.04,
             "Each player counted once per season if they have at least one scraped game log.",
             ha="center", fontsize=14, color="#64748b")
             
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")

def fig_player_retention(season_df: pd.DataFrame, out_path: Path) -> None:
    seasons = sorted(int(s) for s in season_df["season"].dropna().unique())
    if not seasons:
        return
    # Anchor on 2023 to show decreasing bars
    earliest = seasons[0]
    earliest_players = set(season_df.loc[season_df["season"] == earliest, "player_id"])
    if not earliest_players:
        return

    rows = []
    for s in seasons:
        players_s = set(season_df.loc[season_df["season"] == s, "player_id"])
        overlap = len(earliest_players & players_s)
        rows.append((s, overlap))

    fig, ax = plt.subplots(figsize=(10, 7))
    labels = [str(s) for s, _ in rows]
    vals = [v for _, v in rows]
    colors = [HIGHLIGHT if s == earliest else BLUE for s, _ in rows]
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    for rect, val in zip(bars, vals):
        pct = 100.0 * val / len(earliest_players)
        ax.text(rect.get_x() + rect.get_width() / 2, val + max(vals) * 0.01,
                f"{val}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=15,
                fontweight="bold", color="#1e293b")

    ax.set_title(f"Retention of {earliest} Cohort Over Time", fontsize=20, fontweight="bold")
    ax.set_xlabel("Season", fontsize=18)
    ax.set_ylabel(f"Players from {earliest} still active", fontsize=18)
    ax.set_ylim(0, max(vals) * 1.16)
    style_axes(ax)
    fig.text(0.5, -0.04,
             f"Of the {len(earliest_players)} players active in {earliest}, each bar shows how many "
             f"continued to play in subsequent seasons.",
             ha="center", fontsize=14, color="#64748b")
             
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("sim_stats/player_features_train_2023_2026.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "train_seasons" in df.columns:
        df["season"] = df["train_seasons"].astype(str).str.split(",")
        df = df.explode("season")
        df["season"] = df["season"].str.strip().str.replace('"', '')
        
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    
    set_theme()
    fig_active_players_per_season(df, args.out_dir / "active_players_per_season.png")
    fig_player_retention(df, args.out_dir / "player_retention.png")

if __name__ == "__main__":
    main()
