from pathlib import Path
import pandas as pd

ROOT = Path("data/gamelogs")
OUT_REPORT = "duplicate_player_files.csv"


def get_season_from_path(path: Path):
    """
    Try to get season from:
    1. CSV column named 'season'
    2. parent folder name, e.g. data/gamelogs/2026/player.csv
    """
    parent = path.parent.name
    if parent.isdigit():
        return int(parent)
    return None


def read_player_and_season(csv_path: Path):
    try:
        df = pd.read_csv(csv_path, nrows=5, engine="python", on_bad_lines="skip")
    except Exception as e:
        return {
            "file": str(csv_path),
            "player": None,
            "season": get_season_from_path(csv_path),
            "status": f"read_error: {e}",
        }

    if df.empty or len(df.columns) == 0:
        return {
            "file": str(csv_path),
            "player": None,
            "season": get_season_from_path(csv_path),
            "status": "empty_csv",
        }

    # first column = player name, according to your scraper
    first_col = df.columns[0]

    player_values = (
        df[first_col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    if len(player_values) == 0:
        player = None
    else:
        player = player_values.iloc[0]

    # prefer season column if exists
    if "season" in df.columns:
        season_values = pd.to_numeric(df["season"], errors="coerce").dropna()
        if len(season_values) > 0:
            season = int(season_values.iloc[0])
        else:
            season = get_season_from_path(csv_path)
    else:
        season = get_season_from_path(csv_path)

    return {
        "file": str(csv_path),
        "player": player,
        "season": season,
        "status": "ok",
    }


def main():
    csv_files = sorted(ROOT.rglob("*.csv"))

    print(f"Found CSV files: {len(csv_files)}")

    rows = []
    for path in csv_files:
        rows.append(read_player_and_season(path))

    all_files = pd.DataFrame(rows)

    # Save all scanned files too, useful for debugging
    all_files.to_csv("all_gamelog_files_scanned.csv", index=False, encoding="utf-8")

    valid = all_files[
        (all_files["status"] == "ok") &
        (all_files["player"].notna()) &
        (all_files["season"].notna())
    ].copy()

    # normalize only for comparison
    valid["player_norm"] = (
        valid["player"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("*", "", regex=False)
    )

    duplicate_keys = (
        valid
        .groupby(["player_norm", "season"])
        .filter(lambda g: len(g) > 1)
        .copy()
    )

    if duplicate_keys.empty:
        print("No duplicate player-season files found.")
        duplicate_keys.to_csv(OUT_REPORT, index=False, encoding="utf-8")
        return

    duplicate_keys = duplicate_keys.sort_values(["player_norm", "season", "file"])

    duplicate_keys.to_csv(OUT_REPORT, index=False, encoding="utf-8")

    print()
    print(f"Duplicate player-season groups found: {duplicate_keys.groupby(['player_norm', 'season']).ngroups}")
    print(f"Duplicate file rows written to: {OUT_REPORT}")
    print()

    for (player_norm, season), group in duplicate_keys.groupby(["player_norm", "season"]):
        display_name = group["player"].iloc[0]

        print("=" * 80)
        print(f"PLAYER: {display_name}")
        print(f"SEASON: {season}")
        print(f"FILES:")

        for file_path in group["file"]:
            print(f"  - {file_path}")

    print()
    print("Done.")
    print(f"Full report: {OUT_REPORT}")
    print("All scanned files: all_gamelog_files_scanned.csv")


if __name__ == "__main__":
    main()