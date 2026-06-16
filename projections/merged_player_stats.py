import os
from glob import glob
import pandas as pd
from season_stats import open_file, create_stats


def collect_all_csvs(base_dir="data/gamelogs", seasons=None):
    if seasons is None:
        seasons = ["2023", "2024", "2025"]

    player_files = {}

    for season in seasons:
        season_path = os.path.join(base_dir, season, "*.csv")
        for path in glob(season_path):
            player_id = os.path.basename(path).replace(".csv", "")
            if player_id not in player_files:
                player_files[player_id] = []
            player_files[player_id].append((season, path))

    return player_files


def collect_player_stats(base_dir="data/gamelogs", seasons=None):
    players_stats = {}
    player_files = collect_all_csvs(base_dir, seasons)

    for pid, files in player_files.items():
        stats_list = []
        for season, path in files:
            df = open_file(path)
            stats = create_stats(df, path, season)
            stats_list.append(stats)
        players_stats[pid] = stats_list

    return players_stats


def flatten_stats(players_stats):
    rows = []

    for player_id, stats_list in players_stats.items():
        for season_stats in stats_list:
            row = {
                "player_id": player_id,
                "season": season_stats.get("season"),
            }

            for stat_name, stat_data in season_stats.items():
                if stat_name in ["player_id", "season"]:
                    continue

                if isinstance(stat_data, dict):
                    for key, value in stat_data.items():
                        row[f"{stat_name}_{key}"] = value
                else:
                    row[stat_name] = stat_data

            rows.append(row)

    return pd.DataFrame(rows)


def main():
    all_stats = collect_player_stats(
        base_dir="data/gamelogs",
        seasons=["2023", "2024", "2025", "2026"]
    )

    out_df = flatten_stats(all_stats)


    out_path = "sim_stats/player_season_stats.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Saved {len(out_df)} player-season rows to {out_path}")
    print(out_df.head())


if __name__ == "__main__":
    main()