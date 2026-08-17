from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


PRED_CSV = Path("sim_stats/h2h_value_2026.csv")
ACTUAL_CSV = Path("sim_stats/h2h_value_actual_2026.csv")
NAME_CSV = Path("web_scraper/player_gamelog_urls.csv")
DEFAULT_OUT = Path("plots/rank_vs_actual_rank_2026.png")


def load_player_names(csv_path: Path) -> pd.Series:
    urls = pd.read_csv(csv_path)
    urls["player_id"] = urls["gamelog_url"].str.extract(
        r"/players/[a-z]/([^/]+)/gamelog", expand=False
    )
    return (
        urls.dropna(subset=["player_id"])
        .drop_duplicates("player_id")
        .set_index("player_id")["player"]
    )


def short_name(full_name: str) -> str:
    parts = full_name.strip().split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {parts[-1]}"
    return full_name


def plot_rank_scatter(
    df: pd.DataFrame,
    out_path: Path,
    n_label_misses: int = 10,
    top_n: int = 30,
):
    df = df.dropna(subset=["actual_rank_H2H"]).copy()
    df["actual_rank_H2H"] = pd.to_numeric(df["actual_rank_H2H"])
    
    # Calculate lim before filling
    max_actual = int(df["actual_rank_H2H"].max())
    max_pred = int(pd.to_numeric(df["pred_rank_H2H"]).max()) if not df["pred_rank_H2H"].isna().all() else 0
    max_rank = max(max_actual, max_pred)
    pad = max(5, int(max_rank * 0.03))
    lim = max_rank + pad

    # Fill missing predicted ranks with a value near lim so they show up as missed
    df["pred_rank_H2H"] = pd.to_numeric(df["pred_rank_H2H"]).fillna(lim - 10)
    
    df["rank_abs_error"] = (df["pred_rank_H2H"] - df["actual_rank_H2H"]).abs()

    df["actual_top"] = df["actual_rank_H2H"] <= top_n
    df["pred_top"] = df["pred_rank_H2H"] <= top_n

    both_top = df[df["actual_top"] & df["pred_top"]]
    missed = df[df["actual_top"] & ~df["pred_top"]]
    overrated = df[~df["actual_top"] & df["pred_top"]]
    rest = df[~df["actual_top"] & ~df["pred_top"]]

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(
        rest["actual_rank_H2H"],
        rest["pred_rank_H2H"],
        s=28,
        alpha=0.4,
        color="#7f8c8d",
        edgecolors="none",
        label="Other players",
        zorder=2,
    )
    ax.scatter(
        both_top["actual_rank_H2H"],
        both_top["pred_rank_H2H"],
        s=80,
        alpha=0.9,
        color="#27ae60",
        edgecolors="#1e5631",
        linewidths=0.7,
        label=f"Top {top_n} in both ({len(both_top)})",
        zorder=5,
    )
    ax.scatter(
        missed["actual_rank_H2H"],
        missed["pred_rank_H2H"],
        s=80,
        alpha=0.9,
        color="#e67e22",
        edgecolors="#a04000",
        linewidths=0.7,
        label=f"Missed — actual top {top_n} ({len(missed)})",
        zorder=4,
    )
    ax.scatter(
        overrated["actual_rank_H2H"],
        overrated["pred_rank_H2H"],
        s=80,
        alpha=0.9,
        color="#e74c3c",
        edgecolors="#922b21",
        linewidths=0.7,
        label=f"Overrated — projected top {top_n} ({len(overrated)})",
        zorder=4,
    )

    ax.axvline(top_n, color="#27ae60", linestyle=":", linewidth=0.9, alpha=0.5, zorder=1)
    ax.axhline(top_n, color="#27ae60", linestyle=":", linewidth=0.9, alpha=0.5, zorder=1)

    ax.plot([0, lim], [0, lim], color="#2c3e50", linestyle="--", linewidth=1.2, label="y = x", zorder=1)

    rank_mae = df["rank_abs_error"].mean()
    rank_bias = (df["pred_rank_H2H"] - df["actual_rank_H2H"]).mean()
    bias_txt = "over-ranked" if rank_bias < 0 else "under-ranked"
    hit_rate = len(both_top) / top_n if top_n else 0
    
    stats_text = (
        f"n = {len(df)}\n"
        f"Mean |rank error| = {rank_mae:.1f}\n"
        f"Bias = {rank_bias:+.1f} ({bias_txt})\n"
        f"Top {top_n} recall: {len(both_top)}/{top_n} ({100 * hit_rate:.0f}%)"
    )
    ax.text(
        0.97,
        0.97,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=15,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="white", alpha=0.9, edgecolor="#cbd5e1"),
    )

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Actual rank (2026 H2H)", fontsize=12)
    ax.set_ylabel("Projected rank (2026 H2H)", fontsize=12)
    ax.set_title("Projected vs Actual Player Rank — 2026", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Scatter plot of projected rank vs actual rank (2026 H2H evaluation)."
    )
    parser.add_argument("--pred-csv", type=Path, default=PRED_CSV)
    parser.add_argument("--actual-csv", type=Path, default=ACTUAL_CSV)
    parser.add_argument("--name-csv", type=Path, default=NAME_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--label-misses", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    pred = pd.read_csv(args.pred_csv)
    actual = pd.read_csv(args.actual_csv)
    
    # Merge using an outer join so we keep actual players even if they weren't predicted
    df = actual[["player_id", "rank_H2H"]].rename(columns={"rank_H2H": "actual_rank_H2H"}).merge(
        pred[["player_id", "rank_H2H"]].rename(columns={"rank_H2H": "pred_rank_H2H"}),
        on="player_id",
        how="outer"
    )

    if args.name_csv.exists():
        names = load_player_names(args.name_csv)
        df["player_name"] = df["player_id"].map(names).fillna(df["player_id"])
    else:
        df["player_name"] = df["player_id"]

    plot_rank_scatter(df, args.out, n_label_misses=args.label_misses, top_n=args.top_n)


if __name__ == "__main__":
    main()
