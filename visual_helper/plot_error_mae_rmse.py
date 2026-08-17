from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUMMARY_CSV = Path("sim_stats/evaluation_2026_summary.csv")
PRED_CSV = Path("sim_stats/projected_2026_weekly.csv")
ACTUAL_CSV = Path("sim_stats/actual_2026_weekly.csv")
DEFAULT_OUT = Path("plots/error_mae_rmse_normalized.png")

STAT_LABELS = {
    "PTS": "Points",
    "REB": "Rebounds",
    "AST": "Assists",
    "STL": "Steals",
    "BLK": "Blocks",
    "TOV": "Turnovers",
    "FG3M": "3-Pointers\nMade",
    "FG%": "Field Goal\n%",
    "FT%": "Free Throw\n%",
    "games_played": "Games\nPlayed",
}


def actual_column(stat: str) -> str:
    if stat == "games_played":
        return "games_played_mean"
    return f"{stat}_mean"


def load_normalized_errors(
    summary_path: Path,
    pred_path: Path,
    actual_path: Path,
) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    pred = pd.read_csv(pred_path)
    actual = pd.read_csv(actual_path)

    merged = pred.merge(actual, on="player_id", how="inner", suffixes=("_pred", "_actual"))

    rows = []
    for _, row in summary.iterrows():
        stat = row["stat"]
        acol = actual_column(stat)
        actual_col = f"{acol}_actual" if f"{acol}_actual" in merged.columns else acol
        actual_vals = pd.to_numeric(merged[actual_col], errors="coerce").dropna()
        sigma = actual_vals.std()
        if sigma == 0 or np.isnan(sigma):
            continue

        rows.append(
            {
                "stat": stat,
                "label": STAT_LABELS.get(stat, stat),
                "mae": row["mae"],
                "rmse": row["rmse"],
                "sigma_actual": sigma,
                "norm_mae": row["mae"] / sigma,
                "norm_rmse": row["rmse"] / sigma,
                "n": row["n"],
            }
        )

    return pd.DataFrame(rows)


def plot_normalized_mae_rmse(df: pd.DataFrame, out_path: Path):
    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(13, 6.5))

    ax.bar(
        x - width / 2,
        df["norm_mae"],
        width,
        label="Mean Absolute Error ÷ σ",
        color="#4a90d9",
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar(
        x + width / 2,
        df["norm_rmse"],
        width,
        label="Root Mean Squared Error ÷ σ",
        color="#e67e22",
        edgecolor="white",
        linewidth=0.6,
    )

    ax.axhline(1.0, color="#95a5a6", linestyle="--", linewidth=1.0, alpha=0.8, label="1σ reference")

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=0, ha="center", fontsize=10)
    ax.set_ylabel("Normalized error (÷ std. dev. of actual 2026)", fontsize=12)
    ax.set_title(
        "Projection Error per Statistic — Normalized MAE & RMSE",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, max(df["norm_rmse"].max() * 1.12, 1.05))

    easiest = df.nsmallest(2, "norm_mae")["label"].str.replace("\n", " ").tolist()
    hardest = df.nlargest(2, "norm_rmse")["label"].str.replace("\n", " ").tolist()
    ax.text(
        0.98,
        0.98,
        f"Easier: {', '.join(easiest)}\nHarder: {', '.join(hardest)}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="#bdc3c7"),
    )

    fig.text(
        0.5,
        0.01,
        "σ = standard deviation of each stat's actual 2026 season totals across "
        f"{int(df['n'].iloc[0])} evaluated players. "
        "Lower bars = easier to predict.",
        ha="center",
        fontsize=9,
        color="#555555",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Normalized MAE/RMSE bar chart per statistic (2026 evaluation)."
    )
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--pred-csv", type=Path, default=PRED_CSV)
    parser.add_argument("--actual-csv", type=Path, default=ACTUAL_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = load_normalized_errors(args.summary_csv, args.pred_csv, args.actual_csv)
    plot_normalized_mae_rmse(df, args.out)


if __name__ == "__main__":
    main()
