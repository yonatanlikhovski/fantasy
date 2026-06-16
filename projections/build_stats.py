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

OUT_PATH = Path("sim_stats") / "player_features_train_2023_2026.csv"

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

SEASON_WEIGHTS = {
    "2023": 0.10,
    "2024": 0.20,
    "2025": 0.30,
    "2026": 0.40,
}


LEAGUE_AVG_DURABILITY = 65
DURABILITY_REGRESSION = 0.15

COUNTING_STD_MULT = 1.15
PERCENT_STD_MULT = 1.25
ROLE_STD_MULT = 1.20
ROLE_MEAN_BLEND = 0.70
ROLE_TREND_STRENGTH = 0.25

def binomial_std(p: float, attempts_mean: float) -> float:
    """Std of a proportion for ~Binomial with mean attempts per game."""
    attempts_mean = max(1.0, float(attempts_mean))
    v = p * (1 - p) / attempts_mean
    return float(math.sqrt(v))

def numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
def role_proxy(df: pd.DataFrame) -> pd.Series:
    """
    A minutes-free approximation of player role / opportunity.

    Higher value means the player was more involved in the game.
    """
    fga = numeric_col(df, "fga")
    fta = numeric_col(df, "fta")
    ast = numeric_col(df, "ast")
    tov = numeric_col(df, "tov")
    reb = numeric_col(df, "trb")
    stl = numeric_col(df, "stl")
    blk = numeric_col(df, "blk")

    role = (
        fga
        + 0.44 * fta
        + ast
        + tov
        + 0.5 * reb
        + 2.0 * stl
        + 2.0 * blk
    )

    return role.clip(lower=0.0)

def per_player_aggregate(paths: List[Tuple[str, str]]) -> Dict[str, float]:
    """
    Aggregate player seasons using recency-weighted season summaries.
    Gives more weight to recent seasons and uses a role/opportunity proxy.
    """
    season_data = {}

    for season, p in paths:
        season = str(season)
        df = open_file(p)

        if df is None or df.empty:
            continue

        df = df.rename(columns={c: c.lower() for c in df.columns})
        gp = len(df)

        if gp == 0:
            continue

        rec = {"gp": gp}

        # Role / opportunity proxy
        role = role_proxy(df)
        rec["ROLE_mean"] = float(role.mean())
        rec["ROLE_std"] = float(role.std(ddof=1)) if len(role) > 1 else 0.0

        total_role = float(role.sum())

        # Counting stats and shooting volume stats
        for stat in COUNTING + ["FGA", "FTA"]:
            src = SRC_MAP[stat]
            s = numeric_col(df, src)

            rec[f"{stat}_mean"] = float(s.mean())
            rec[f"{stat}_std"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0

            if total_role > 0:
                rec[f"{stat}_per_role"] = float(s.sum() / total_role)
            else:
                rec[f"{stat}_per_role"] = 0.0

        # Shooting totals — outside the stat loop
        fgm = float(numeric_col(df, "fgm").sum())
        fga = float(numeric_col(df, "fga").sum())
        ftm = float(numeric_col(df, "ftm").sum())
        fta = float(numeric_col(df, "fta").sum())

        rec["FGM_total"] = fgm
        rec["FGA_total"] = fga
        rec["FTM_total"] = ftm
        rec["FTA_total"] = fta

        rec["FGP_mean"] = float(fgm / fga) if fga > 0 else 0.0
        rec["FTP_mean"] = float(ftm / fta) if fta > 0 else 0.0

        # Important: save this season summary inside the season loop
        season_data[season] = rec

    if not season_data:
        raise ValueError("No games found for player")

    raw_weights = {
        season: SEASON_WEIGHTS.get(season, 1.0)
        for season in season_data.keys()
    }

    weight_sum = sum(raw_weights.values())
    weights = {
        season: raw_weights[season] / weight_sum
        for season in raw_weights
    }

    feat: Dict[str, float] = {}

    def weighted_mean(key: str) -> float:
        return float(sum(weights[s] * season_data[s][key] for s in season_data))

    def weighted_std(mean_key: str, std_key: str, mult: float = 1.0) -> float:
        mu = weighted_mean(mean_key)

        var = sum(
            weights[s] * (
                season_data[s][std_key] ** 2
                + (season_data[s][mean_key] - mu) ** 2
            )
            for s in season_data
        )

        return float(math.sqrt(max(0.0, var)) * mult)

    # Role / opportunity projection
    feat["ROLE_mean"] = weighted_mean("ROLE_mean")
    feat["ROLE_std"] = weighted_std(
        "ROLE_mean",
        "ROLE_std",
        mult=ROLE_STD_MULT
    )

    available_seasons = sorted(season_data.keys())

    if len(available_seasons) >= 2:
        prev_season = available_seasons[-2]
        last_season = available_seasons[-1]

        role_delta = (
            season_data[last_season]["ROLE_mean"]
            - season_data[prev_season]["ROLE_mean"]
        )

        feat["ROLE_mean"] = feat["ROLE_mean"] + ROLE_TREND_STRENGTH * role_delta
        feat["ROLE_mean"] = max(0.0, feat["ROLE_mean"])

    # Counting stats
    for stat in COUNTING:
        raw_mean = weighted_mean(f"{stat}_mean")
        per_role = weighted_mean(f"{stat}_per_role")
        role_mean = feat["ROLE_mean"] * per_role

        feat[f"{stat}_per_role"] = per_role
        feat[f"{stat}_mean"] = (
            ROLE_MEAN_BLEND * role_mean
            + (1 - ROLE_MEAN_BLEND) * raw_mean
        )

        feat[f"{stat}_std"] = weighted_std(
            f"{stat}_mean",
            f"{stat}_std",
            mult=COUNTING_STD_MULT
        )

    # Shooting volumes
    for stat in ["FGA", "FTA"]:
        raw_mean = weighted_mean(f"{stat}_mean")
        per_role = weighted_mean(f"{stat}_per_role")
        role_mean = feat["ROLE_mean"] * per_role

        feat[f"{stat}_per_role"] = per_role
        feat[f"{stat}_mean"] = (
            ROLE_MEAN_BLEND * role_mean
            + (1 - ROLE_MEAN_BLEND) * raw_mean
        )

        feat[f"{stat}_std"] = weighted_std(
            f"{stat}_mean",
            f"{stat}_std",
            mult=COUNTING_STD_MULT
        )

    # FG% weighted by recent shooting volume
    fga_weighted = sum(
        weights[s] * season_data[s]["FGA_mean"]
        for s in season_data
    )

    if fga_weighted > 0:
        fgp_mean = sum(
            weights[s] * season_data[s]["FGP_mean"] * season_data[s]["FGA_mean"]
            for s in season_data
        ) / fga_weighted
    else:
        fgp_mean = 0.0

    # FT% weighted by recent shooting volume
    fta_weighted = sum(
        weights[s] * season_data[s]["FTA_mean"]
        for s in season_data
    )

    if fta_weighted > 0:
        ftp_mean = sum(
            weights[s] * season_data[s]["FTP_mean"] * season_data[s]["FTA_mean"]
            for s in season_data
        ) / fta_weighted
    else:
        ftp_mean = 0.0

    feat["FGP_mean"] = float(np.clip(fgp_mean, 0, 1))
    feat["FTP_mean"] = float(np.clip(ftp_mean, 0, 1))

    # Shooting uncertainty
    fgp_between_var = sum(
        weights[s] * (season_data[s]["FGP_mean"] - feat["FGP_mean"]) ** 2
        for s in season_data
    )

    ftp_between_var = sum(
        weights[s] * (season_data[s]["FTP_mean"] - feat["FTP_mean"]) ** 2
        for s in season_data
    )

    feat["FGP_std"] = float(
        math.sqrt(
            binomial_std(feat["FGP_mean"], feat["FGA_mean"]) ** 2
            + fgp_between_var
        )
        * PERCENT_STD_MULT
    )

    feat["FTP_std"] = float(
        math.sqrt(
            binomial_std(feat["FTP_mean"], feat["FTA_mean"]) ** 2
            + ftp_between_var
        )
        * PERCENT_STD_MULT
    )

    # Durability
    weighted_gp = sum(
        weights[s] * season_data[s]["gp"]
        for s in season_data
    )

    durability = (
        (1 - DURABILITY_REGRESSION) * weighted_gp
        + DURABILITY_REGRESSION * LEAGUE_AVG_DURABILITY
    )

    gp_var = sum(
        weights[s] * (season_data[s]["gp"] - weighted_gp) ** 2
        for s in season_data
    )

    durability_std = max(8.0, math.sqrt(max(0.0, gp_var)))

    durability = int(round(durability))
    durability = max(0, min(82, durability))

    feat["durability"] = durability
    feat["durability_std"] = float(durability_std)

    return feat

def main(seasons: List[str] | None = None):
    # Collect player -> [(season, path), ...]
    if seasons is None:
        seasons = ["2023", "2024", "2025","2026"]

    # Collect player -> [(season, path), ...]
    player_files = collect_all_csvs(seasons=seasons)
    rows = []

    for pid, paths in player_files.items():
        try:
            feat = per_player_aggregate(paths)
            row = {
                "player_id": pid,
                "durability": feat["durability"],
                "durability_std": feat.get("durability_std", 8.0),
                    
                "ROLE_mean": feat["ROLE_mean"],
                "ROLE_std": feat["ROLE_std"],

                "PTS_mean": feat["PTS_mean"], "PTS_std": feat["PTS_std"],
                "REB_mean": feat["REB_mean"], "REB_std": feat["REB_std"],
                "AST_mean": feat["AST_mean"], "AST_std": feat["AST_std"],
                "STL_mean": feat["STL_mean"], "STL_std": feat["STL_std"],
                "BLK_mean": feat["BLK_mean"], "BLK_std": feat["BLK_std"],
                "TOV_mean": feat["TOV_mean"], "TOV_std": feat["TOV_std"],

                "FG3M_mean": feat["FG3M_mean"], "FG3M_std": feat["FG3M_std"],
                "PTS_per_role": feat["PTS_per_role"],
                "REB_per_role": feat["REB_per_role"],
                "AST_per_role": feat["AST_per_role"],
                "STL_per_role": feat["STL_per_role"],
                "BLK_per_role": feat["BLK_per_role"],
                "TOV_per_role": feat["TOV_per_role"],
                "FG3M_per_role": feat["FG3M_per_role"],
                "FGA_per_role": feat["FGA_per_role"],
                "FTA_per_role": feat["FTA_per_role"],

                "FGA_mean": feat["FGA_mean"], "FGA_std": feat["FGA_std"],
                "FGP_mean": feat["FGP_mean"], "FGP_std": feat["FGP_std"],

                "FTA_mean": feat["FTA_mean"], "FTA_std": feat["FTA_std"],
                "FTP_mean": feat["FTP_mean"], "FTP_std": feat["FTP_std"],
                "source": "nba_history",
                "train_seasons": ",".join(seasons),
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
    main()