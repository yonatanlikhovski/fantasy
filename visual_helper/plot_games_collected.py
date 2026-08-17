import argparse
from pathlib import Path
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
        "axes.titlesize": 20,
        "axes.titleweight": "bold",
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "font.family": "DejaVu Sans",
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/gamelogs"))
    parser.add_argument("--out-path", type=Path, default=Path("plots/games_collected.png"))
    args = parser.parse_args()

    seasons = ["2023", "2024", "2025", "2026"]
    games_per_season = []

    for season in seasons:
        season_dir = args.data_dir / season
        total_games = 0
        if season_dir.exists():
            for csv_file in season_dir.glob("*.csv"):
                with open(csv_file, "r", encoding="utf-8") as f:
                    lines = sum(1 for _ in f)
                    if lines > 1:
                        total_games += (lines - 1)
        games_per_season.append(total_games)

    set_theme()
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(seasons, games_per_season, color=PALETTE[0], width=0.62)
    
    for rect, val in zip(bars, games_per_season):
        ax.text(rect.get_x() + rect.get_width() / 2, val + max(games_per_season) * 0.01,
                f"{int(val):,}", ha="center", va="bottom", fontsize=15, fontweight="bold",
                color="#1e293b")

    ax.set_title("Total Games Collected Per Season", fontweight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Total Game Logs (Records)")
    ax.set_ylim(0, max(games_per_season) * 1.15 if max(games_per_season) > 0 else 100)
    
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_path, bbox_inches="tight")
    print(f"Wrote {args.out_path}")

if __name__ == "__main__":
    main()
