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

OUT_PATH = Path("sim_stats") / "player_features_train_2023_2025.csv"

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
    "2023": 0.15,
    "2024": 0.25,
    "2025": 0.5
}

LEAGUE_AVG_DURABILITY = 65
DURABILITY_REGRESSION = 0.15

COUNTING_STD_MULT = 1.15
PERCENT_STD_MULT = 1.25
ROLE_STD_MULT = 1.20
ROLE_MEAN_BLEND = 0.70
ROLE_TREND_STRENGTH = 0.25

YOUNG_GAMES_CUTOFF = 250
OLD_GAMES_CUTOFF = 900

YOUNG_MAX_GROWTH = 0.10
OLD_MAX_DECLINE = 0.12
OLD_DECLINE_SPAN = 600


def binomial_std(p: float, attempts_mean: float) -> float:
    """Std of a proportion for ~Binomial with mean attempts per game."""
    attempts_mean = max(1.0, float(attempts_mean))
    v = p * (1 - p) / attempts_mean
    return float(math.sqrt(v))


def numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def extract_player_name(df: pd.DataFrame, fallback: str) -> str:
    """
    Extract the real player name from inside the gamelog CSV.

    If no name column exists, fallback to the filename-based id.
    """
    name_candidates = [
        "player",
        "player_name",
        "name",
        "player name",
        "Player",
        "Player Name",
        "PLAYER",
    ]

    cmap = {c.lower().strip(): c for c in df.columns}

    for cand in name_candidates:
        key = cmap.get(cand.lower().strip())
        if key is None:
            continue

        names = df[key].dropna().astype(str).str.strip()
        names = names[names != ""]

        if not names.empty:
            # Remove a trailing Basketball-Reference style star if it exists.
            return names.iloc[0].rstrip("*").strip()

    return fallback


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


def extract_career_games(df: pd.DataFrame) -> float | None:
    """
    Extract career games from gcar if it exists.
    If missing, return None.
    """
    if "gcar" not in df.columns:
        return None

    gcar = pd.to_numeric(df["gcar"], errors="coerce").dropna()

    if gcar.empty:
        return None

    val = float(gcar.max())

    if val <= 0:
        return None

    return val


def games_played_trajectory_factor(career_games: float) -> tuple[str, float]:
    """
    Convert career games into a role adjustment factor.

    Young players: slight growth boost.
    Prime players: stable.
    Old players: gradual decline.
    """
    if career_games <= YOUNG_GAMES_CUTOFF:
        stage = "young_growth"

        # 0 games -> +10%, 250 games -> +0%
        progress = career_games / YOUNG_GAMES_CUTOFF
        factor = 1.0 + YOUNG_MAX_GROWTH * (1.0 - progress)

    elif career_games <= OLD_GAMES_CUTOFF:
        stage = "prime_stable"
        factor = 1.0

    else:
        stage = "decline"

        # 900 games -> 0% decline
        # 1500 games -> max decline
        decline_progress = min(
            1.0,
            (career_games - OLD_GAMES_CUTOFF) / OLD_DECLINE_SPAN
        )

        factor = 1.0 - OLD_MAX_DECLINE * decline_progress

    return stage, float(factor)


def per_player_aggregate(paths: List[Tuple[str, str]]) -> Dict:
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

        # Keep old filename id only as debug information.
        source_file_id = Path(p).stem

        # Extract real player name from inside the CSV before lowercasing columns.
        player_name = extract_player_name(df, fallback=source_file_id)

        # Now normalize columns for calculations.
        df = df.rename(columns={c: c.lower().strip() for c in df.columns})
        gp = len(df)

        if gp == 0:
            continue

        career_games = extract_career_games(df)

        rec = {
            "gp": gp,
            "career_games": career_games,
            "player_name": player_name,
            "source_file_id": source_file_id,
        }

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

        # Shooting totals
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

    feat: Dict = {}

    # Use the most recent season's real name.
    latest_season = sorted(season_data.keys())[-1]
    feat["player_name"] = season_data[latest_season]["player_name"]
    feat["source_file_id"] = season_data[latest_season]["source_file_id"]

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

    # Recent role trend adjustment
    if len(available_seasons) >= 2:
        prev_season = available_seasons[-2]
        last_season = available_seasons[-1]

        role_delta = (
            season_data[last_season]["ROLE_mean"]
            - season_data[prev_season]["ROLE_mean"]
        )

        feat["ROLE_mean"] = feat["ROLE_mean"] + ROLE_TREND_STRENGTH * role_delta
        feat["ROLE_mean"] = max(0.0, feat["ROLE_mean"])

    # Career-games trajectory adjustment
    career_games_candidates = [
        season_data[s].get("career_games")
        for s in season_data
        if season_data[s].get("career_games") is not None
    ]

    if career_games_candidates:
        career_games = max(career_games_candidates)
    else:
        # fallback: only games we have in our dataset
        career_games = sum(season_data[s]["gp"] for s in season_data)

    trajectory_stage, trajectory_factor = games_played_trajectory_factor(career_games)

    feat["career_games"] = float(career_games)
    feat["trajectory_stage"] = trajectory_stage
    feat["trajectory_factor"] = float(trajectory_factor)

    feat["ROLE_mean_before_trajectory"] = float(feat["ROLE_mean"])
    feat["ROLE_mean"] = max(0.0, feat["ROLE_mean"] * trajectory_factor)

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
    if seasons is None:
        seasons = ["2023", "2024", "2025"]

    player_files = collect_all_csvs(seasons=seasons)
    rows = []

    for pid, paths in player_files.items():
        try:
            feat = per_player_aggregate(paths)

            row = {
                # From now on, player_id is the real player name.
                "player_id": feat.get("player_name", pid),

                # Keep old filename id for debugging only.
                "source_file_id": feat.get("source_file_id", pid),

                "durability": feat["durability"],
                "durability_std": feat.get("durability_std", 8.0),

                "ROLE_mean": feat["ROLE_mean"],
                "career_games": feat.get("career_games", np.nan),
                "trajectory_stage": feat.get("trajectory_stage", "unknown"),
                "trajectory_factor": feat.get("trajectory_factor", 1.0),
                "ROLE_mean_before_trajectory": feat.get(
                    "ROLE_mean_before_trajectory",
                    feat["ROLE_mean"]
                ),
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
            print(f"[skip] {pid}: {e}")

    out = pd.DataFrame(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"✅ Wrote {len(out)} players → {OUT_PATH}")

    if "player_id" in out.columns:
        dupes = out[out["player_id"].duplicated(keep=False)]
        if not dupes.empty:
            print("\n⚠️ Duplicate player names found:")
            print(dupes[["player_id", "source_file_id"]].sort_values("player_id").to_string(index=False))


if __name__ == "__main__":
    main()