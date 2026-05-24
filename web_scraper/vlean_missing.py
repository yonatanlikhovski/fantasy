import re
import pandas as pd
from pathlib import Path

# INPUTS
MISSING_LOG = "missing_now.csv"       # or "failed_gamelogs.csv"
DONE_LOG = "done_gamelogs.csv"

# The folders where successful gamelog csvs may exist.
# maunal.py writes to data/manual_gamelogs by default.
SUCCESS_DIRS = [
    Path("data/gamelogs"),
    Path("data/manual_gamelogs"),
]

# OUTPUTS
OUT_MISSING = "missing_now_clean.csv"
OUT_RETRY = "incremented.csv"
OUT_ALREADY_DONE = "already_done_removed.csv"


def norm_name(x):
    return re.sub(r"\s+", " ", str(x).replace("*", "").strip().lower())


def parse_url(url):
    """
    Return (player_id, season) from a BBR gamelog URL.
    Example:
    https://www.basketball-reference.com/players/g/greenjo01/gamelog/2026
    -> ("greenjo01", 2026)
    """
    m = re.search(r"/players/[a-z]/([^/]+)/gamelog/(\d{4})", str(url), re.I)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def id_family(player_id):
    """
    Convert greenjo01 -> greenjo.
    This lets us treat greenjo01, greenjo02, greenjo03 as the same guessed-player family.
    """
    if player_id is None:
        return None
    return re.sub(r"\d{2}$", "", str(player_id))


def possible_ids(player_id, max_suffix=9):
    """
    For greenjo01, return greenjo01 ... greenjo09.
    This catches cases where original URL failed but greenjo02 worked.
    """
    fam = id_family(player_id)
    if not fam:
        return []
    return [f"{fam}{i:02d}" for i in range(1, max_suffix + 1)]


def file_exists_success(player_id, season):
    """
    Check if a successful csv exists on disk, either in main gamelogs or manual_gamelogs.
    """
    if player_id is None or pd.isna(season):
        return False

    season = int(season)

    for pid in possible_ids(player_id):
        for root in SUCCESS_DIRS:
            path = root / str(season) / f"{pid}.csv"
            if path.exists() and path.stat().st_size > 0:
                return True

    return False


def build_done_sets():
    """
    Build success sets from done_gamelogs.csv.
    Also include player_id family so greenjo01 can match greenjo02 if needed.
    """
    done_name_season = set()
    done_id_season = set()
    done_family_season = set()
    done_url_set = set()

    if not Path(DONE_LOG).exists():
        return done_name_season, done_id_season, done_family_season, done_url_set

    done = pd.read_csv(DONE_LOG, engine="python", on_bad_lines="skip")

    if "gamelog_url" in done.columns:
        done_url_set = set(done["gamelog_url"].dropna().astype(str))

    if "player" in done.columns and "season" in done.columns:
        for _, row in done.iterrows():
            season = pd.to_numeric(row.get("season"), errors="coerce")
            if pd.notna(season):
                done_name_season.add((norm_name(row.get("player")), int(season)))

    # player_id may be present as a column, but also parse from gamelog_url
    for _, row in done.iterrows():
        season = pd.to_numeric(row.get("season"), errors="coerce")
        if pd.isna(season):
            _, url_season = parse_url(row.get("gamelog_url", ""))
            season = url_season

        if pd.isna(season) or season is None:
            continue

        player_id = row.get("player_id", None)
        if pd.isna(player_id) if player_id is not None else True:
            player_id, _ = parse_url(row.get("gamelog_url", ""))

        if player_id:
            done_id_season.add((str(player_id), int(season)))
            done_family_season.add((id_family(str(player_id)), int(season)))

    return done_name_season, done_id_season, done_family_season, done_url_set


def main():
    missing = pd.read_csv(MISSING_LOG, engine="python", on_bad_lines="skip")

    needed = {"player", "season", "gamelog_url"}
    missing_cols = needed - set(missing.columns)
    if missing_cols:
        raise ValueError(f"{MISSING_LOG} is missing columns: {missing_cols}")

    done_name_season, done_id_season, done_family_season, done_url_set = build_done_sets()

    rows = []
    reasons = []

    for _, row in missing.iterrows():
        player = norm_name(row["player"])
        season = pd.to_numeric(row["season"], errors="coerce")
        url = str(row["gamelog_url"])
        url_player_id, url_season = parse_url(url)

        if pd.isna(season) and url_season is not None:
            season = url_season

        if pd.isna(season):
            already_done = False
            reason = "cannot-parse-season"
        else:
            season = int(season)
            fam = id_family(url_player_id)

            checks = {
                "exact_url_in_done": url in done_url_set,
                "player_season_in_done": (player, season) in done_name_season,
                "player_id_season_in_done": (url_player_id, season) in done_id_season,
                "player_id_family_season_in_done": (fam, season) in done_family_season,
                "csv_file_exists": file_exists_success(url_player_id, season),
            }

            already_done = any(checks.values())
            reason = ",".join(k for k, v in checks.items() if v) if already_done else ""

        rows.append(already_done)
        reasons.append(reason)

    missing["already_done"] = rows
    missing["done_reason"] = reasons

    already_done_df = missing[missing["already_done"]].copy()
    still_missing = missing[~missing["already_done"]].copy()

    # Remove helper columns from retry file
    retry = still_missing[["player", "season", "gamelog_url"]].drop_duplicates()

    already_done_df.to_csv(OUT_ALREADY_DONE, index=False, encoding="utf-8")
    still_missing.to_csv(OUT_MISSING, index=False, encoding="utf-8")
    retry.to_csv(OUT_RETRY, index=False, encoding="utf-8")

    print(f"Input rows: {len(missing)}")
    print(f"Removed because already successful: {len(already_done_df)}")
    print(f"Still missing: {len(still_missing)}")
    print()
    print(f"Wrote removed rows: {OUT_ALREADY_DONE}")
    print(f"Wrote clean missing file: {OUT_MISSING}")
    print(f"Wrote retry file for maunal.py: {OUT_RETRY}")

    if len(already_done_df):
        print()
        print("Examples removed:")
        print(already_done_df[["player", "season", "gamelog_url", "done_reason"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
