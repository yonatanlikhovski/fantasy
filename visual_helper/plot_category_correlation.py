import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

H2H_CATS = ["PTS", "REB", "AST", "STL", "BLK", "FG3M", "TOV", "FG", "FT"]

STAT_LABELS = {
    "PTS": "Points",
    "REB": "Rebounds",
    "AST": "Assists",
    "STL": "Steals",
    "BLK": "Blocks",
    "FG3M": "3-Pointers Made",
    "TOV": "Turnovers",
    "FG": "Field Goal %",
    "FT": "Free Throw %",
    "FG%": "Field Goal %",
    "FT%": "Free Throw %",
    "FG_pct": "Field Goal %",
    "FT_pct": "Free Throw %",
    "games_played": "Games Played",
}

def pretty(stat: str) -> str:
    full = STAT_LABELS.get(stat)
    if full is None:
        return stat
    abbr = stat.replace("_pct", "%")
    return f"{full}\n({abbr})"

def fig_category_correlation(value_df: pd.DataFrame, out_path: Path) -> None:
    cats = [c for c in H2H_CATS if c in value_df.columns]
    if len(cats) < 2:
        print("  [SKIP] not enough category columns for correlation")
        return
    data = value_df[cats].apply(pd.to_numeric, errors="coerce")
    corr = data.corr().values

    # Using the theme defined in original script
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1",
        "axes.titleweight": "bold",
        "font.family": "DejaVu Sans",
    })

    fig, ax = plt.subplots(figsize=(10, 9))
    DIVERGING = plt.get_cmap("RdBu_r")
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1, aspect="equal")
    
    labels = [pretty(c) for c in cats]
    ax.set_xticks(range(len(cats)))
    ax.set_yticks(range(len(cats)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(labels, fontsize=8.5)
    
    for i in range(len(cats)):
        for j in range(len(cats)):
            v = corr[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.6 else "#0f172a", fontsize=9)
            
    ax.set_title("Correlation between scoring categories across players")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation (red = positive, blue = negative)")
    ax.grid(False)
    fig.text(0.5, -0.02,
             "Pearson correlation between players' normalized category scores. Red = move together, "
             "blue = move oppositely, near-white = unrelated.",
             ha="center", fontsize=9, color="#64748b")
             
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote plot -> {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("sim_stats/h2h_value_2027.csv"), help="Path to H2H value CSV")
    parser.add_argument("--out", type=Path, default=Path("plots/category_correlation_heatmap.png"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    fig_category_correlation(df, args.out)

if __name__ == "__main__":
    main()
