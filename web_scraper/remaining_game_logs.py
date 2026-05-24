import pandas as pd
from pathlib import Path

ALL_URLS = "player_gamelog_urls.csv"
DONE_LOG = "done_gamelogs.csv"
OUT = "incremented.csv"

def read_done_urls():
    if not Path(DONE_LOG).exists():
        return set()

    # robust because your done_gamelogs.csv may have inconsistent columns
    try:
        done = pd.read_csv(DONE_LOG, usecols=["gamelog_url"])
    except Exception:
        done = pd.read_csv(DONE_LOG, engine="python", on_bad_lines="skip")
        if "gamelog_url" not in done.columns:
            return set()

    return set(done["gamelog_url"].dropna().astype(str))

def main():
    all_urls = pd.read_csv(ALL_URLS)

    needed_cols = ["player", "season", "gamelog_url"]
    all_urls = all_urls[needed_cols].dropna(subset=["gamelog_url"]).drop_duplicates()

    done_urls = read_done_urls()

    missing = all_urls[~all_urls["gamelog_url"].astype(str).isin(done_urls)].copy()

    missing.to_csv(OUT, index=False, encoding="utf-8")

    print(f"Total expected gamelogs: {len(all_urls)}")
    print(f"Already done: {len(done_urls)}")
    print(f"Missing written to {OUT}: {len(missing)}")
    print(missing.head(20))

if __name__ == "__main__":
    main()