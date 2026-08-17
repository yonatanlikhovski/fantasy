import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

def set_theme() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#555555",
        "axes.linewidth": 1.0,
        "axes.axisbelow": True,
        "font.family": "DejaVu Sans",
    })

def main():
    set_theme()
    csv_path = Path("sim_stats/strategic_draft_standings_2027.csv")
    df = pd.read_csv(csv_path)
    
    # Sort by W ascending for horizontal bar chart
    df = df.sort_values("W", ascending=True).reset_index(drop=True)
    
    labels = []
    for _, row in df.iterrows():
        strat = row["strategy"]
        punts = row["punts"]
        risk = row["risk"]
        
        if strat == "baseline":
            name = "Balanced"
        elif strat == "blocks_heavy":
            name = "Block-focused"
        elif strat == "guard":
            name = "Guard build"
        elif strat == "big":
            name = "Big build"
        else:
            name = strat
            
        if risk > 0.1:
            name += "\n(conservative)"
            
        if pd.isna(punts) or not str(punts).strip():
            punt_text = "(no punt)"
        else:
            punt_str = str(punts).replace(",", "/")
            punt_text = f"(ignoring {punt_str})"
            
        if "\n" in name:
            labels.append(f"{name}\n{punt_text}")
        else:
            labels.append(f"{name}\n{punt_text}")
        
    # Green for > 4.5, red for < 4.5
    colors = ["#C73D35" if w < 4.5 else "#549942" for w in df["W"]]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(range(len(df)), df["W"], color=colors, height=0.65, edgecolor="white")
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=9)
    
    # Draw vertical line for League Average
    ax.axvline(x=4.5, color="#555555", linestyle="--", alpha=0.8, label="League avg (4.5 W)")
    
    ax.set_xlabel("Expected wins (250 simulated seasons)", fontsize=10)
    
    # X-axis grid only
    ax.xaxis.grid(True, linestyle="-", alpha=0.3, color="#cccccc")
    ax.yaxis.grid(False)
    
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 8.5)
    
    out_path = Path("plots/expected_wins_strategy.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Wrote plot to {out_path}")

if __name__ == "__main__":
    main()
