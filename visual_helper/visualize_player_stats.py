from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEFAULT_INPUT = Path("sim_stats/player_season_stats.csv")
DEFAULT_OUT_DIR = Path("visual_helper/plots")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_stats(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV file: {csv_path}")

    df = pd.read_csv(csv_path)

    if "season" not in df.columns:
        raise ValueError("CSV must contain a 'season' column")

    if "player_id" not in df.columns:
        raise ValueError("CSV must contain a 'player_id' column")

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    return df


def print_available_stats(df: pd.DataFrame):
    print("\nAvailable columns:")
    print("-" * 80)
    for col in df.columns:
        print(col)
    print("-" * 80)


def ensure_out_dir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def clean_stat_series(df: pd.DataFrame, stat_col: str) -> pd.Series:
    if stat_col not in df.columns:
        raise ValueError(f"Column '{stat_col}' does not exist in the CSV")

    return pd.to_numeric(df[stat_col], errors="coerce").dropna()


# ---------------------------------------------------------------------
# Plot 1: distribution of any stat
# ---------------------------------------------------------------------

def plot_distribution(df: pd.DataFrame, stat_col: str, out_dir: Path):
    values = clean_stat_series(df, stat_col)

    if values.empty:
        print(f"[SKIP] No numeric values found for {stat_col}")
        return

    plt.figure(figsize=(9, 6))
    plt.hist(values, bins=30, edgecolor="black")
    plt.title(f"Distribution of {stat_col}")
    plt.xlabel(stat_col)
    plt.ylabel("Number of player-seasons")
    plt.grid(alpha=0.3)

    out_path = out_dir / f"distribution_{stat_col}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[OK] Saved distribution plot: {out_path}")


# ---------------------------------------------------------------------
# Plot 2: number of player stat records by year
# ---------------------------------------------------------------------

def plot_number_of_player_stats_by_year(df: pd.DataFrame, out_dir: Path):
    counts = (
        df
        .dropna(subset=["season"])
        .groupby("season")
        .size()
        .reset_index(name="num_player_stat_rows")
    )

    plt.figure(figsize=(9, 6))
    plt.bar(counts["season"].astype(str), counts["num_player_stat_rows"])
    plt.title("Number of Player Stat Rows by Season")
    plt.xlabel("Season")
    plt.ylabel("Number of player-season rows")
    plt.grid(axis="y", alpha=0.3)

    out_path = out_dir / "num_player_stats_by_year.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[OK] Saved player-count-by-year plot: {out_path}")
    print()
    print("Number of player stat rows by year:")
    print(counts.to_string(index=False))


# ---------------------------------------------------------------------
# Plot 3: year-by-year averages for selected stats
# ---------------------------------------------------------------------

def plot_average_stat_by_year(df: pd.DataFrame, stat_col: str, out_dir: Path):
    if stat_col not in df.columns:
        print(f"[SKIP] Missing column: {stat_col}")
        return None

    temp = df[["season", stat_col]].copy()
    temp[stat_col] = pd.to_numeric(temp[stat_col], errors="coerce")
    temp = temp.dropna(subset=["season", stat_col])

    if temp.empty:
        print(f"[SKIP] No numeric data for {stat_col}")
        return None

    yearly = (
        temp
        .groupby("season")[stat_col]
        .mean()
        .reset_index(name=f"avg_{stat_col}")
    )

    four_year_avg = temp[stat_col].mean()

    plt.figure(figsize=(9, 6))
    plt.plot(yearly["season"].astype(str), yearly[f"avg_{stat_col}"], marker="o")
    plt.axhline(four_year_avg, linestyle="--", label=f"4-year avg = {four_year_avg:.3f}")

    plt.title(f"Average {stat_col} by Season")
    plt.xlabel("Season")
    plt.ylabel(f"Average {stat_col}")
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = out_dir / f"avg_{stat_col}_by_year.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[OK] Saved yearly average plot for {stat_col}: {out_path}")

    yearly["four_year_avg"] = four_year_avg
    return yearly


# ---------------------------------------------------------------------
# Plot 4: combined table of yearly averages
# ---------------------------------------------------------------------

def create_summary_table(df: pd.DataFrame, stat_cols: list[str], out_dir: Path):
    summary = None
    four_year_rows = []

    for stat_col in stat_cols:
        if stat_col not in df.columns:
            print(f"[SKIP] Missing summary column: {stat_col}")
            continue

        temp = df[["season", stat_col]].copy()
        temp[stat_col] = pd.to_numeric(temp[stat_col], errors="coerce")
        temp = temp.dropna(subset=["season", stat_col])

        if temp.empty:
            continue

        yearly = (
            temp
            .groupby("season")[stat_col]
            .mean()
            .reset_index()
            .rename(columns={stat_col: f"avg_{stat_col}"})
        )

        if summary is None:
            summary = yearly
        else:
            summary = summary.merge(yearly, on="season", how="outer")

        four_year_rows.append({
            "stat": stat_col,
            "four_year_avg": temp[stat_col].mean()
        })

    if summary is not None:
        summary = summary.sort_values("season")
        summary_path = out_dir / "yearly_average_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"[OK] Saved yearly average summary CSV: {summary_path}")
        print()
        print(summary.to_string(index=False))

    four_year_df = pd.DataFrame(four_year_rows)
    four_year_path = out_dir / "four_year_average_summary.csv"
    four_year_df.to_csv(four_year_path, index=False)
    print(f"[OK] Saved 4-year average summary CSV: {four_year_path}")
    print()
    print(four_year_df.to_string(index=False))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize player season stats from stats/player_season_stats.csv"
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to player_season_stats.csv"
    )

    parser.add_argument(
        "--stat",
        type=str,
        default=None,
        help="Column name to plot as a distribution, for example pts_mean"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="Print available columns and exit"
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = DEFAULT_OUT_DIR

    ensure_out_dir(out_dir)

    df = load_stats(csv_path)

    print(f"Loaded rows: {len(df)}")
    print(f"Loaded columns: {len(df.columns)}")

    if args.list:
        print_available_stats(df)
        return

    # Always create this first
    plot_number_of_player_stats_by_year(df, out_dir)

    # Default key plots
    default_stats = [
        "FG%_mean",
        "pts_mean",
        "Gcar_final",
    ]

    print()
    print("Creating default yearly average plots...")
    for stat_col in default_stats:
        plot_average_stat_by_year(df, stat_col, out_dir)

    print()
    print("Creating summary CSVs...")
    create_summary_table(df, default_stats, out_dir)

    # Optional custom distribution
    if args.stat is not None:
        print()
        print(f"Creating custom distribution for: {args.stat}")
        plot_distribution(df, args.stat, out_dir)

    print()
    print("Done.")
    print(f"Plots saved in: {out_dir}")


if __name__ == "__main__":
    main()