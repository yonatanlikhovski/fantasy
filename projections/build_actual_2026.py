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

COUNTING = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FGM", "FGA", "FTM", "FTA"]


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

        for _, path in paths:
            df = open_file(path)
            if df is None or df.empty:
                continue
            df = df.rename(columns={c: c.lower() for c in df.columns})
            frames.append(df)

        if not frames:
            continue

        df = pd.concat(frames, ignore_index=True)

        row = {
            "player_id": player_id,
            "season": season,
            "games_played_mean": len(df),
        }

        totals = {}

        for stat in COUNTING:
            src = SRC_MAP[stat]
            if src in df.columns:
                total = float(pd.to_numeric(df[src], errors="coerce").fillna(0).sum())
            else:
                total = 0.0

            totals[stat] = total
            row[f"{stat}_mean"] = total / weekly_scale

        row["FG%_mean"] = totals["FGM"] / totals["FGA"] if totals["FGA"] > 0 else np.nan
        row["FT%_mean"] = totals["FTM"] / totals["FTA"] if totals["FTA"] > 0 else np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = Path("sim_stats") / "actual_2026_weekly.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"✅ Wrote actual 2026 weekly stats → {out_path}")


if __name__ == "__main__":
    main()