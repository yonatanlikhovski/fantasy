from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_INPUT = Path("sim_stats/player_season_stats.csv")
DEFAULT_OUT_DIR = Path("visual_helper/plots/distributions_by_season")


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


def list_columns(df: pd.DataFrame):
    print("\nAvailable columns:")
    print("-" * 80)
    for col in df.columns:
        print(col)
    print("-" * 80)


def plot_distribution_by_season(
    df: pd.DataFrame,
    stat_col: str,
    out_dir: Path,
    rounded: bool = True,
):
    if stat_col not in df.columns:
        raise ValueError(f"Column '{stat_col}' does not exist. Use --list to see columns.")

    temp = df[["season", "player_id", stat_col]].copy()
    temp[stat_col] = pd.to_numeric(temp[stat_col], errors="coerce")
    temp = temp.dropna(subset=["season", stat_col])

    if temp.empty:
        print(f"No numeric data found for {stat_col}")
        return

    if rounded:
        plot_col = f"{stat_col}_rounded"
        temp[plot_col] = temp[stat_col].round().astype(int)
    else:
        plot_col = stat_col

    out_dir.mkdir(parents=True, exist_ok=True)

    seasons = sorted(temp["season"].dropna().unique())

    for season in seasons:
        season_df = temp[temp["season"] == season].copy()
        values = season_df[plot_col].dropna()

        if values.empty:
            continue

        plt.figure(figsize=(9, 6))

        if rounded:
            min_val = int(values.min())
            max_val = int(values.max())

            # integer-centered bins
            bins = range(min_val, max_val + 2)

            plt.hist(values, bins=bins, edgecolor="black", align="left")
            plt.xticks(range(min_val, max_val + 1))

            xlabel = f"Rounded {stat_col}"
        else:
            plt.hist(values, bins=30, edgecolor="black")
            xlabel = stat_col

        plt.title(f"Distribution of {stat_col} by Player — {season}")
        plt.xlabel(xlabel)
        plt.ylabel("Number of players")
        plt.grid(axis="y", alpha=0.3)

        out_path = out_dir / f"{stat_col}_{season}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"[OK] {season}: saved {out_path}")

    print()
    print(f"Done. Plots saved in: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Create per-season player distribution graphs for one stat."
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to stats/player_season_stats.csv",
    )

    parser.add_argument(
        "--stat",
        type=str,
        default="pts_mean",
        help="Column to plot, for example pts_mean, trb_mean, ast_mean, Gcar_final",
    )

    parser.add_argument(
        "--no-round",
        action="store_true",
        help="Do not round values before plotting",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="Print available columns and exit",
    )

    args = parser.parse_args()

    df = load_stats(Path(args.csv))

    if args.list:
        list_columns(df)
        return

    plot_distribution_by_season(
        df=df,
        stat_col=args.stat,
        out_dir=DEFAULT_OUT_DIR,
        rounded=not args.no_round,
    )


if __name__ == "__main__":
    main()