import pandas as pd
import re

def create_valid_years(row):
    base_url = row["url"].replace(".html", "")
    start_year = max(int(row["From"]), 2023)   # minimum 2023
    end_year = int(row["To"]) if not pd.isna(row["To"]) else 2025  # default if missing

    if end_year < 2023:
        return []   # no valid seasons

    valid_urls = []
    for season in range(start_year, end_year + 1):
        url = f"{base_url}/gamelog/{season}"
        valid_urls.append({
            "player": row["Player"],
            "season": season,
            "gamelog_url": url
        })
    return valid_urls


if __name__ == "__main__":
    df = pd.read_csv("player_urls.csv", encoding="utf-8")

    all_rows = []
    for _, row in df.iterrows():
        all_rows.extend(create_valid_years(row))

    gamelog_df = pd.DataFrame(all_rows)
    gamelog_df.to_csv("player_gamelog_urls.csv", index=False, encoding="utf-8")
