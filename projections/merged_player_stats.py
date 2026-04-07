import os
from glob import glob
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

if __name__ == "__main__":
    all_stats = collect_player_stats()
    # show first player as a test
    first_pid = next(iter(all_stats))
    print(f"Stats for {first_pid}:")
    for s in all_stats[first_pid]:
        print(s["season"], "->", s["pts"])
