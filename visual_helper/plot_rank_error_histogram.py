from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRED_CSV = Path("sim_stats/h2h_value_2026.csv")
ACTUAL_CSV = Path("sim_stats/h2h_value_actual_2026.csv")
DEFAULT_OUT = Path("plots/rank_error_histogram_2026.png")


def plot_rank_error_histogram(
    df: pd.DataFrame,
    out_path: Path,
    bin_width: int = 10,
):
    df = df.dropna(subset=["rank_abs_error"]).copy()
    errors = pd.to_numeric(df["rank_abs_error"], errors="coerce").dropna()

    max_err = int(errors.max())
    bins = np.arange(0, max_err + bin_width, bin_width)

    fig, ax = plt.subplots(figsize=(10, 6))

    counts, edges, patches = ax.hist(
        errors,
        bins=bins,
        color="#7f8c8d",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.85,
        label="Players",
    )

    for patch, left, right in zip(patches, edges[:-1], edges[1:]):
        if left >= 10 and right <= 40:
            patch.set_facecolor("#e67e22")
            patch.set_alpha(0.9)

    median = errors.median()
    mean = errors.mean()
    ax.axvline(median, color="#2c3e50", linestyle="--", linewidth=1.4, label=f"Median = {median:.0f}")
    ax.axvline(mean, color="#3498db", linestyle=":", linewidth=1.4, label=f"Mean = {mean:.0f}")

    in_10_40 = ((errors >= 10) & (errors <= 40)).sum()
    tail_100 = (errors > 100).sum()
    ax.text(
        0.98,
        0.98,
        f"n = {len(errors)}  |  10–40 error: {in_10_40} players ({100 * in_10_40 / len(errors):.0f}%)\n"
        f">100 error: {tail_100} players ({100 * tail_100 / len(errors):.0f}%)",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="#bdc3c7"),
    )

    ax.set_xlabel("Absolute rank error", fontsize=12)
    ax.set_ylabel("Number of players", fontsize=12)
    ax.set_title("Distribution of Absolute Rank Error — 2026 H2H", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.82), framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_xlim(0, max_err + bin_width)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Histogram of absolute rank error (2026 H2H evaluation)."
    )
    parser.add_argument("--pred-csv", type=Path, default=PRED_CSV)
    parser.add_argument("--actual-csv", type=Path, default=ACTUAL_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bin-width", type=int, default=10)
    args = parser.parse_args()

    pred = pd.read_csv(args.pred_csv)
    actual = pd.read_csv(args.actual_csv)
    
    # Merge using an outer join
    df = actual[["player_id", "rank_H2H"]].rename(columns={"rank_H2H": "actual_rank_H2H"}).merge(
        pred[["player_id", "rank_H2H"]].rename(columns={"rank_H2H": "pred_rank_H2H"}),
        on="player_id",
        how="outer"
    )

    # To calculate rank absolute error, we need to handle NaN for pred_rank_H2H
    # We assign a high rank (miss) for players without predictions
    max_actual = int(df["actual_rank_H2H"].max())
    max_pred = int(pd.to_numeric(df["pred_rank_H2H"]).max()) if not df["pred_rank_H2H"].isna().all() else 0
    max_rank = max(max_actual, max_pred)
    pad = max(5, int(max_rank * 0.03))
    lim = max_rank + pad
    
    df["pred_rank_H2H"] = pd.to_numeric(df["pred_rank_H2H"]).fillna(lim - 10)
    df["actual_rank_H2H"] = pd.to_numeric(df["actual_rank_H2H"]).fillna(lim - 10)

    if "rank_abs_error" not in df.columns:
        df["rank_abs_error"] = (df["pred_rank_H2H"] - df["actual_rank_H2H"]).abs()

    plot_rank_error_histogram(df, args.out, bin_width=args.bin_width)


if __name__ == "__main__":
    main()
