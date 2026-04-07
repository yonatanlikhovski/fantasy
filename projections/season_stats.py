import pandas as pd
import os

M_STATS = ['fgm', 'fga','fg3m', 'fg3a', 'ftm', 'fta',
           'trb', 'ast', 'stl', 'blk', 'tov', 'PF', 'pts','GmSc']
P_STATS = ['FG%','FT%']
S_STATS = ['Rk', 'Gcar']

def open_file(path):
    return pd.read_csv(path)

def create_stats(df, path=None, season=None):
    stats = {}

    # --- player info ---
    if path:
        stats["player_id"] = os.path.basename(path).replace(".csv", "")
    if season:
        stats["season"] = season

    # --- counting stats ---
    for stat in M_STATS:
        if stat in df.columns:
            total = df[stat].sum().item()
            mean = df[stat].mean().item()
            std = df[stat].std().item()
            std_percent = (std / mean * 100) if mean != 0 else None
            consistency = (mean / std) if std != 0 else None

            stats[stat] = {
                "total": total,
                "mean": mean,
                "std": std,
                "std_percent": std_percent,
                "consistency": consistency
            }

    # --- percentage stats ---
    for stat in P_STATS:
        if stat in df.columns:
            mean = df[stat].mean().item()
            std = df[stat].std().item()
            std_percent = (std / mean * 100) if mean != 0 else None
            consistency = (mean / std) if std != 0 else None

            stats[stat] = {
                "mean": mean,
                "std": std,
                "std_percent": std_percent,
                "consistency": consistency
            }

    # --- season stats ---
    for stat in S_STATS:
        if stat in df.columns and not df.empty:
            last_val = df[stat].iloc[-1].item()
            stats[stat] = {"final": last_val}

    return stats
