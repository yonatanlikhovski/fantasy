from playwright.sync_api import sync_playwright
from urllib.parse import urljoin
import pandas as pd
import random, string, time

BASE_URL = "https://www.basketball-reference.com/players/"  
UAs = [

    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]

def random_user_agent() -> str:
    return random.choice(UAs)

def create_links(base: str):
    return [urljoin(base, f"{ch}/") for ch in string.ascii_lowercase]

def open_page(url: str, browser) -> pd.DataFrame:
    # New context per request so UA is applied cleanly
    context = browser.new_context(
        user_agent=random_user_agent(),
        extra_http_headers={
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.basketball-reference.com/players/",
        },
        viewport={"width": 1366, "height": 900},
    )
    page = context.new_page()
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_selector("#players", timeout=15000)

    # Parse the already-loaded HTML (pandas won’t do its own fetch)
    html = page.content()
    df = pd.read_html(html, attrs={"id": "players"})[0]

    # Optional: keep only active players (To == 2025)
    if "To" in df.columns:
        df = df[pd.to_numeric(df["To"], errors="coerce") == 2026]

    page.close()
    context.close()
    return df

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    urls = create_links(BASE_URL)
    active_players = []

    for url in urls:
        # Gentle jitter to look less bot-like
        time.sleep(random.uniform(2.0, 4.5))
        try:
            active_players.append(open_page(url, browser))
        except Exception as e:
            print(f"Skipping {url}: {e}")

    if active_players:
        all_active = pd.concat(active_players, ignore_index=True)
        print(all_active.head())
        print("Total rows:", all_active.shape[0])

    all_active.to_csv("active_players.csv")

    browser.close()

if __name__ == "__main__":
    with sync_playwright() as p:
        run(p)
