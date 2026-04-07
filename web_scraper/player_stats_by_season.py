import os
import re
import time
import random
import pandas as pd
from pathlib import Path
from playwright.sync_api import sync_playwright

# ---------- Config ----------
OUT_ROOT = Path("data/gamelogs")
DONE_LOG = Path("done_gamelogs.csv")
FAIL_LOG = Path("failed_gamelogs.csv")

UAs = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]
def random_user_agent() -> str:
    return random.choice(UAs)

# ---------- Helpers ----------
slug_re = re.compile(r"/players/([a-z])/([a-z0-9]+)/gamelog/(\d{4})/?$", re.I)

def parse_meta_from_url(url: str):
    m = slug_re.search(url)
    if not m:
        raise ValueError(f"Cannot parse (letter, player_id, season) from URL: {url}")
    letter, player_id, season = m.group(1), m.group(2), int(m.group(3))
    return letter, player_id, season

def out_path_for(player_id: str, season: int) -> Path:
    return OUT_ROOT / f"{season}" / f"{player_id}.csv"

def append_log(path: Path, row: dict):
    # Append (create with header if needed)
    exists = path.exists()
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not exists, index=False, encoding="utf-8")

def load_done_set() -> set[str]:
    if DONE_LOG.exists():
        return set(pd.read_csv(DONE_LOG, usecols=["gamelog_url"])["gamelog_url"].tolist())
    return set()

def is_written(player_id: str, season: int) -> bool:
    return out_path_for(player_id, season).exists()

# ---------- Core scraping ----------
def fetch_gamelog_df(page_html: str) -> pd.DataFrame:
    """
    Try to read the regular-season gamelog table.
    Primary: id='pgl_basic'
    Fallbacks: some pages may expose a single 'player_game_log_reg' table
    """
    # Try primary
    try:
        return pd.read_html(page_html, attrs={"id": "pgl_basic"})[0]
    except ValueError:
        pass
    # Fallback
    try:
        return pd.read_html(page_html, attrs={"id": "player_game_log_reg"})[0]
    except ValueError:
        raise ValueError("No gamelog table found (pgl_basic / player_game_log_reg)")

def clean_gamelog(df: pd.DataFrame) -> pd.DataFrame:
    # Drop repeated header rows inside tbody
    if "Rk" in df.columns:
        df = df[df["Rk"] != "Rk"].copy()

    # Coerce numeric where appropriate
    for col in df.columns:
        if col not in {"Rk","G","Date","Tm","Opp","Result","Unnamed: 6","GS","Notes"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rename common stats (optional)
    rename_map = {
        "Tm": "team", "Opp": "opponent", "Date": "date",
        "MP": "minutes", "PTS": "pts", "AST": "ast", "TRB": "trb",
        "STL": "stl", "BLK": "blk", "TOV": "tov",
        "FG": "fgm", "FGA": "fga", "3P": "fg3m", "3PA": "fg3a",
        "FT": "ftm", "FTA": "fta", "ORB": "orb", "DRB": "drb",
        "+/-": "plus_minus",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Filter to rows that look like actual games (Gcar exists on season logs)
    if "Gcar" in df.columns:
        df = df.dropna(subset=["Gcar"])

    # Normalize date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype("string")

    return df.reset_index(drop=True)

def scrape_one_gamelog(browser, url: str) -> pd.DataFrame:
    context = browser.new_context(user_agent=random_user_agent())
    page = context.new_page()
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        # Wait a bit to be polite / ensure render
        page.wait_for_timeout(500)
        html = page.content()
        df = fetch_gamelog_df(html)
        df = clean_gamelog(df)
        return df
    finally:
        page.close()
        context.close()

# ---------- Orchestrator ----------
def main():
    # Input URLs: file with columns [player, season, gamelog_url] (from your earlier step)
    urls_df = pd.read_csv("incremented.csv", encoding="utf-8")

    # Ensure output root exists
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    done_urls = load_done_set()
    missing_rows = []  # in-memory list; also append to FAIL_LOG as we go

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            # Shuffle to be nice to the site
            urls = urls_df.sample(frac=1.0, random_state=42).to_dict("records")

            for rec in urls:
                url = rec["gamelog_url"]
                player = rec.get("player")
                season = int(rec["season"])

                # Parse player_id & sanity check season from URL
                try:
                    _, player_id, season_from_url = parse_meta_from_url(url)
                except Exception as e:
                    print(f"[PARSE-ERR] {url}: {e}")
                    row = {"gamelog_url": url, "player": player, "season": season, "reason": f"parse:{e}"}
                    append_log(FAIL_LOG, row)
                    continue

                if season != season_from_url:
                    # Trust URL's season
                    season = season_from_url

                out_path = out_path_for(player_id, season)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # Skip if already written or marked done
                if out_path.exists() or url in done_urls:
                    continue

                # Polite jitter
                time.sleep(random.uniform(1.0, 2.5))
                try:
                    df = scrape_one_gamelog(browser, url)
                    # Add meta
                    df.insert(0, "player", player if player is not None else player_id)
                    df.insert(1, "player_id", player_id)
                    df.insert(2, "season", season)

                    # Save CSV for this (player, season)
                    df.to_csv(out_path, index=False, encoding="utf-8")
                    # Mark success
                    append_log(DONE_LOG, {
                        "gamelog_url": url, "player": player, "player_id": player_id,
                        "season": season, "rows": len(df)
                    })
                    print(f"[OK] {player_id} {season} → {out_path} ({len(df)} rows)")
                except Exception as e:
                    print(f"[FAIL] {url}: {e}")
                    row = {"gamelog_url": url, "player": player, "season": season, "reason": str(e)}
                    append_log(FAIL_LOG, row)
                    missing_rows.append(row)
        finally:
            browser.close()

    # Also dump current missing list to a quick CSV for convenience (optional)
    if missing_rows:
        pd.DataFrame(missing_rows).to_csv("missing_now.csv", index=False, encoding="utf-8")
        print(f"Missing this run: {len(missing_rows)} (see missing_now.csv)")

if __name__ == "__main__":
    main()
