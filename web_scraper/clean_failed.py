import pandas as pd
from pathlib import Path

DONE_LOG = "done_gamelogs.csv"

# Use whichever file currently contains your failed/missing rows:
# If you want to clean failed_gamelogs.csv, change this to "failed_gamelogs.csv"
MISSING_LOG = "missing_now.csv"

OUT_MISSING = "missing_now_clean.csv"
OUT_RETRY = "incremented.csv"


def norm_player(x):
    return str(x).strip().lower().replace("*", "")


def main():
    done = pd.read_csv(DONE_LOG, engine="python", on_bad_lines="skip")
    missing = pd.read_csv(MISSING_LOG, engine="python", on_bad_lines="skip")

    done["player_norm"] = done["player"].map(norm_player)
    missing["player_norm"] = missing["player"].map(norm_player)

    done["season_int"] = pd.to_numeric(done["season"], errors="coerce").astype("Int64")
    missing["season_int"] = pd.to_numeric(missing["season"], errors="coerce").astype("Int64")

    done_pairs = set(
        (player, int(season))
        for player, season in zip(done["player_norm"], done["season_int"])
        if pd.notna(season)
    )

    def is_already_done(row):
        if pd.isna(row["season_int"]):
            return False
        return (row["player_norm"], int(row["season_int"])) in done_pairs

    already_done_mask = missing.apply(is_already_done, axis=1)

    still_missing = missing[~already_done_mask].copy()

    # remove helper columns
    still_missing = still_missing.drop(columns=["player_norm", "season_int"], errors="ignore")

    # remove duplicates
    still_missing = still_missing.drop_duplicates(subset=["player", "season", "gamelog_url"])

    still_missing.to_csv(OUT_MISSING, index=False, encoding="utf-8")

    # this is the file maunal.py expects
    retry_cols = ["player", "season", "gamelog_url"]
    still_missing[retry_cols].to_csv(OUT_RETRY, index=False, encoding="utf-8")

    print(f"Rows in missing file: {len(missing)}")
    print(f"Already successful: {already_done_mask.sum()}")
    print(f"Still missing: {len(still_missing)}")
    print(f"Wrote: {OUT_MISSING}")
    print(f"Wrote retry file: {OUT_RETRY}")


if __name__ == "__main__":
    main()