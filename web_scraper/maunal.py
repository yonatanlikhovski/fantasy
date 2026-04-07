# manual_retry.py
import re
import time
import random
import pandas as pd
from io import StringIO
from pathlib import Path
from playwright.sync_api import sync_playwright

# -------- config --------
IN_CSV = "incremented.csv"                 # player, season, gamelog_url
OUT_ROOT = Path("data/manual_gamelogs")    # manual outputs (kept separate from main)
DONE_LOG = Path("done_gamelogs.csv")       # reuse same logs
FAIL_LOG = Path("failed_gamelogs.csv")

UAs = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]
def ua(): return random.choice(UAs)

slug_re = re.compile(r"/players/([a-z])/([a-z0-9]+)/(?:gamelog/(\d{4}))/?$", re.I)

def parse_meta(url: str):
    m = slug_re.search(url)
    if not m:
        raise ValueError(f"Bad gamelog URL: {url}")
    letter, player_id, season = m.group(1), m.group(2), int(m.group(3))
    return letter, player_id, season

def out_path(player_id: str, season: int) -> Path:
    return OUT_ROOT / f"{season}" / f"{player_id}.csv"

def append_log(path: Path, row: dict):
    exists = path.exists()
    pd.DataFrame([row]).to_csv(path, mode="a", header=not exists, index=False, encoding="utf-8")

def bump_suffix_once(url: str) -> str:
    # increment the two-digit suffix right before /gamelog/YYYY
    def repl(m):
        n = int(m.group(1)) + 1
        return f"{n:02d}" + m.group(2)
    return re.sub(r"(\d{2})(/gamelog/\d{4})/?$", repl, url)

def try_suffixes(browser, url: str, max_tries: int = 5) -> tuple[pd.DataFrame, str] | tuple[None, None]:
    """
    Try the original URL, then 4 increments (01->02->...).
    Return (df, working_url) or (None, None)
    """
    trial_url = url
    for attempt in range(max_tries):
        try:
            df = scrape_gamelog(browser, trial_url)
            if df is not None and not df.empty:
                return df, trial_url
        except Exception as e:
            # print for visibility; we'll log outside
            print(f"  attempt {attempt+1} failed: {trial_url} -> {e}")
        # bump for next try
        trial_url = bump_suffix_once(trial_url)
        time.sleep(random.uniform(0.6, 1.2))
    return None, None

def fetch_table(page_html: str) -> pd.DataFrame:
    # Wrap html in StringIO per pandas warning
    buf = StringIO(page_html)
    try:
        return pd.read_html(buf, attrs={"id": "pgl_basic"})[0]
    except ValueError:
        pass
    buf.seek(0)
    return pd.read_html(buf, attrs={"id": "player_game_log_reg"})[0]

def clean(df: pd.DataFrame) -> pd.DataFrame:
    if "Rk" in df.columns:
        df = df[df["Rk"] != "Rk"].copy()
    for col in df.columns:
        if col not in {"Rk","G","Date","Tm","Opp","Result","Unnamed: 6","GS","Notes"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Gcar" in df.columns:
        df = df.dropna(subset=["Gcar"])
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date.astype("string")
    # light normalize names (optional)
    rename = {"Tm":"team","Opp":"opponent","Date":"date","MP":"minutes","PTS":"pts","AST":"ast","TRB":"trb",
              "STL":"stl","BLK":"blk","TOV":"tov","FG":"fgm","FGA":"fga","3P":"fg3m","3PA":"fg3a",
              "FT":"ftm","FTA":"fta","ORB":"orb","DRB":"drb","+/-":"plus_minus"}
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
    return df.reset_index(drop=True)

def scrape_gamelog(browser, url: str) -> pd.DataFrame:
    ctx = browser.new_context(user_agent=ua())
    page = ctx.new_page()
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(400)
        html = page.content()
        df = fetch_table(html)
        df = clean(df)
        return df
    finally:
        page.close()
        ctx.close()

def main():
    IN = pd.read_csv(IN_CSV, encoding="utf-8")
    IN = IN[["player","season","gamelog_url"]].dropna(subset=["gamelog_url"]).drop_duplicates()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            records = IN.to_dict("records")
            # Don’t shuffle here; you said you want a manual sweep
            for rec in records:
                url = rec["gamelog_url"]
                player = rec.get("player")
                season = int(rec.get("season"))
                try:
                    _, player_id, season_from_url = parse_meta(url)
                except Exception as e:
                    print(f"[PARSE] {url} -> {e}")
                    append_log(FAIL_LOG, {"gamelog_url": url, "player": player, "season": season, "reason": f"parse:{e}"})
                    continue
                # trust URL season if mismatched
                season = season_from_url

                outp = out_path(player_id, season)
                outp.parent.mkdir(parents=True, exist_ok=True)

                print(f"[TRY] {player or player_id} {season} …")
                df, working = try_suffixes(browser, url, max_tries=5)
                if df is not None and not df.empty:
                    # add meta
                    df.insert(0, "player", player if player else player_id)
                    df.insert(1, "player_id", player_id)
                    df.insert(2, "season", season)
                    df.to_csv(outp, index=False, encoding="utf-8")
                    append_log(DONE_LOG, {
                        "gamelog_url": working, "player": player, "player_id": player_id,
                        "season": season, "rows": len(df), "mode": "manual"
                    })
                    print(f"[OK]  {player_id} {season} -> {outp} ({len(df)} rows) via {working}")
                else:
                    reason = "no-table-or-empty-after-retries"
                    append_log(FAIL_LOG, {"gamelog_url": url, "player": player,
                                          "season": season, "reason": reason, "mode": "manual"})
                    print(f"[FAIL] {url} -> {reason}")
                # be polite
                time.sleep(random.uniform(0.8, 1.6))
        finally:
            browser.close()

if __name__ == "__main__":
    main()
