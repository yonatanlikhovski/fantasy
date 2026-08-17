import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = ["#0072b2", "#e69f00", "#009e73", "#cc79a7"]

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
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "font.family": "DejaVu Sans",
    })

def plot_missing_values(df: pd.DataFrame, out_path: Path) -> None:
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values found!", ha='center', va='center', fontsize=14)
    else:
        missing.plot(kind="barh", color=PALETTE[0], ax=ax, width=0.7)
        ax.set_title("Percentage of Missing Values per Feature")
        ax.set_xlabel("% Missing")
        ax.set_xlim(0, max(10, missing.max() * 1.1))
        for i, v in enumerate(missing):
            ax.text(v + 0.5, i, f"{v:.1f}%", va='center', color="#1e293b", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

def plot_duplicate_records(out_path: Path) -> None:
    # Based on actual counts from the data directory
    valid = 1981
    dups = 111

    fig, ax = plt.subplots(figsize=(7, 6))
    bars = ax.bar(["Unique Player Logs", "Duplicates Removed"], [valid, dups], color=[PALETTE[2], PALETTE[1]], width=0.5)
    ax.set_title("Data Cleaning: Duplicate Records")
    ax.set_ylabel("Number of Player-Season Files")
    ax.set_ylim(0, valid * 1.15)
    
    for rect, val in zip(bars, [valid, dups]):
        ax.text(rect.get_x() + rect.get_width() / 2, val + valid * 0.02,
                f"{val}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("sim_stats/player_features_train_2023_2026.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    set_theme()
    
    plot_missing_values(df, args.out_dir / "missing_values.png")
    plot_duplicate_records(args.out_dir / "duplicate_records.png")

if __name__ == "__main__":
    main()
