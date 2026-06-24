from pathlib import Path
import numpy as np
import pandas as pd

from merged_player_stats import collect_all_csvs
from season_stats import open_file

SRC_MAP = {
    "PTS": "pts",
    "REB": "trb",
    "AST": "ast",
    "STL": "stl",
    "BLK": "blk",
    "TOV": "tov",
    "FG3M": "fg3m",
    "FGM": "fgm",
    "FGA": "fga",
    "FTM": "ftm",
    "FTA": "fta",
}

COUNTING = [
    "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M",
    "FGM", "FGA", "FTM", "FTA"
]


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
            return names.iloc[0].rstrip("*").strip()

    return fallback


def main():
    season = "2026"
    games_in_season = 82
    games_per_week = 3.5
    weekly_scale = games_in_season / games_per_week

    player_files = collect_all_csvs(
        base_dir="data/gamelogs",
        seasons=[season]
    )

    rows = []

    for player_id, paths in player_files.items():
        frames = []
        real_name = None
        source_file_id = None

        for _, path in paths:
            df = open_file(path)

            if df is None or df.empty:
                continue

            fallback_id = Path(path).stem
            player_name = extract_player_name(df, fallback=fallback_id)

            if real_name is None:
                real_name = player_name
            if source_file_id is None:
                source_file_id = fallback_id

            df = df.rename(columns={c: c.lower().strip() for c in df.columns})
            frames.append(df)

        if not frames:
            continue

        df = pd.concat(frames, ignore_index=True)

        if real_name is None:
            real_name = player_id
        if source_file_id is None:
            source_file_id = player_id

        row = {
            # From now on, player_id is the real player name.
            "player_id": real_name,

            # Keep filename id only for debugging.
            "source_file_id": source_file_id,

            "season": season,
            "games_played_mean": len(df),
        }

        totals = {}

        for stat in COUNTING:
            src = SRC_MAP[stat]

            if src in df.columns:
                total = float(
                    pd.to_numeric(df[src], errors="coerce")
                    .fillna(0)
                    .sum()
                )
            else:
                total = 0.0

            totals[stat] = total
            row[f"{stat}_mean"] = total / weekly_scale

        row["FG%_mean"] = (
            totals["FGM"] / totals["FGA"]
            if totals["FGA"] > 0
            else np.nan
        )

        row["FT%_mean"] = (
            totals["FTM"] / totals["FTA"]
            if totals["FTA"] > 0
            else np.nan
        )

        rows.append(row)

    out = pd.DataFrame(rows)

    out_path = Path("sim_stats") / "actual_2026_weekly.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"✅ Wrote actual 2026 weekly stats → {out_path}")

    if "player_id" in out.columns:
        dupes = out[out["player_id"].duplicated(keep=False)]

        if not dupes.empty:
            print("\n⚠️ Duplicate player names found:")
            print(
                dupes[["player_id", "source_file_id"]]
                .sort_values("player_id")
                .to_string(index=False)
            )


if __name__ == "__main__":
    main()