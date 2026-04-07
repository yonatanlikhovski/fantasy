# build_player_features_summary.py
# Rebuild sim_stats/player_features_summary.csv for the Monte Carlo simulator.
# Uses your multi-season file collector + season file reader.

from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# Your modules
from merged_player_stats import collect_all_csvs
from season_stats import open_file

OUT_PATH = Path("sim_stats") / "player_features_summary.csv"

# Stats we feed into the simulator (per-game)
COUNTING = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]
# Source column mapping from your gamelog CSVs (lowercase)
SRC_MAP = {
    "PTS": "pts",
    "REB": "trb",
    "AST": "ast",
    "STL": "stl",
    "BLK": "blk",
    "TOV": "tov",
    "FG3M": "fg3m",
    # shooting volumes
    "FGM": "fgm",
    "FGA": "fga",
    "FTM": "ftm",
    "FTA": "fta",
}

def binomial_std(p: float, attempts_mean: float) -> float:
    """Std of a proportion for ~Binomial with mean attempts per game."""
    attempts_mean = max(1.0, float(attempts_mean))
    v = p * (1 - p) / attempts_mean
    return float(math.sqrt(v))

def per_player_aggregate(paths: List[Tuple[str, str]]) -> Dict[str, float]:
    """
    Aggregate all seasons for one player.
    Returns a dict of per-game means/stds and durability.
    """
    # Concatenate all games across seasons
    games: List[pd.DataFrame] = []
    total_fgm = total_fga = total_ftm = total_fta = 0.0
    total_gp = 0

    for season, p in paths:
        df = open_file(p)  # user’s helper
        if df is None or df.empty:
            continue
        # Ensure lower-case columns for robustness
        df = df.rename(columns={c: c.lower() for c in df.columns})
        # keep only needed columns that exist
        keep_cols = [SRC_MAP[s] for s in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FGA","FTM","FTA"] if SRC_MAP[s] in df.columns]
        df = df[keep_cols].copy()

        # Totals for % from totals
        total_fgm += float(df.get("fgm", pd.Series(dtype=float)).sum())
        total_fga += float(df.get("fga", pd.Series(dtype=float)).sum())
        total_ftm += float(df.get("ftm", pd.Series(dtype=float)).sum())
        total_fta += float(df.get("fta", pd.Series(dtype=float)).sum())

        total_gp += len(df)
        games.append(df)

    if not games or total_gp == 0:
        raise ValueError("No games found for player")

    allg = pd.concat(games, ignore_index=True)

    # Per-game means & stds for counting stats
    feat: Dict[str, float] = {}
    for stat in COUNTING:
        src = SRC_MAP[stat]
        if src not in allg.columns:
            # fallbacks
            feat[f"{stat}_mean"] = 0.0
            feat[f"{stat}_std"] = 0.5
        else:
            s = allg[src].astype(float)
            feat[f"{stat}_mean"] = float(s.mean())
            # robust std: if 1 game, std = 0
            feat[f"{stat}_std"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0

    # Shooting volumes (per-game mean/std)
    for vol, src in (("FGA", "fga"), ("FTA", "fta")):
        if src in allg.columns:
            s = allg[src].astype(float)
            feat[f"{vol}_mean"] = float(s.mean())
            feat[f"{vol}_std"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        else:
            feat[f"{vol}_mean"] = 0.0
            feat[f"{vol}_std"] = 0.5

    # Percentages from TOTALS across all seasons
    fgp_mean = float(total_fgm / total_fga) if total_fga > 0 else 0.0
    ftp_mean = float(total_ftm / total_fta) if total_fta > 0 else 0.0
    # Std via binomial around per-game attempt means
    fgp_std = binomial_std(fgp_mean, feat["FGA_mean"]) if feat["FGA_mean"] > 0 else 0.05
    ftp_std = binomial_std(ftp_mean, feat["FTA_mean"]) if feat["FTA_mean"] > 0 else 0.05
    feat["FGP_mean"], feat["FGP_std"] = float(np.clip(fgp_mean, 0, 1)), float(fgp_std)
    feat["FTP_mean"], feat["FTP_std"] = float(np.clip(ftp_mean, 0, 1)), float(ftp_std)

    # Durability = average games per season (clipped to [0,82])
    # If we know how many seasons we read:
    #   paths is a list of (season, path) that existed and were non-empty
    # Use count of non-empty seasons to compute per-season GP mean
    seasons_non_empty = max(1, sum(
        1 for _, p in paths
        if (lambda d: d is not None and not d.empty)(open_file(p))
    ))
    durability = int(round(total_gp / seasons_non_empty))
    durability = max(0, min(82, durability))
    feat["durability"] = durability

    return feat

def main(seasons: List[str] | None = None):
    # Collect player -> [(season, path), ...]
    player_files = collect_all_csvs(seasons=seasons)
    rows = []

    for pid, paths in player_files.items():
        try:
            feat = per_player_aggregate(paths)
            row = {
                "player_id": pid,
                "durability": feat["durability"],

                "PTS_mean": feat["PTS_mean"], "PTS_std": feat["PTS_std"],
                "REB_mean": feat["REB_mean"], "REB_std": feat["REB_std"],
                "AST_mean": feat["AST_mean"], "AST_std": feat["AST_std"],
                "STL_mean": feat["STL_mean"], "STL_std": feat["STL_std"],
                "BLK_mean": feat["BLK_mean"], "BLK_std": feat["BLK_std"],
                "TOV_mean": feat["TOV_mean"], "TOV_std": feat["TOV_std"],

                "FG3M_mean": feat["FG3M_mean"], "FG3M_std": feat["FG3M_std"],

                "FGA_mean": feat["FGA_mean"], "FGA_std": feat["FGA_std"],
                "FGP_mean": feat["FGP_mean"], "FGP_std": feat["FGP_std"],

                "FTA_mean": feat["FTA_mean"], "FTA_std": feat["FTA_std"],
                "FTP_mean": feat["FTP_mean"], "FTP_std": feat["FTP_std"],
            }
            rows.append(row)
        except Exception as e:
            # Skip silently but print for debugging; you can log to a file if you like
            print(f"[skip] {pid}: {e}")

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"✅ Wrote {len(out)} players → {OUT_PATH}")

if __name__ == "__main__":
    # By default, uses seasons = ["2023","2024","2025"] via collect_all_csvs
    main()
