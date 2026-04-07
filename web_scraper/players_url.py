import pandas as pd
import re

BASE_URL = "https://www.basketball-reference.com/players/"

def letters_only(s: str) -> str:
    """Keep only a–z (lowercase), drop spaces, hyphens, apostrophes, dots, suffixes."""
    return re.sub(r"[^a-z]", "", s.lower())

def split_name(full: str):
    """Return (first, last). Joins multi-word last names like 'Van Vleet' -> 'VanVleet'."""
    full = full.replace("*", "").strip()
    parts = full.split()
    if len(parts) == 1:
        return parts[0], ""
    # Heuristic: first = first token, last = the rest joined
    first = parts[0]
    last = "".join(parts[1:])
    return first, last

def bbr_url_from_name(full_name: str, seq: int = 1) -> str:
    first, last = split_name(full_name)
    f2 = letters_only(first)[:2]
    l5 = letters_only(last)[:5]
    if not l5 or not f2:
        return ""  # cannot form a proper slug
    first_letter = l5[0]
    nn = f"{seq:02d}"
    return f"{BASE_URL}{first_letter}/{l5}{f2}{nn}.html"

def create_players_urls(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    # assumes a column named 'Player'
    urls = df["Player"].apply(lambda name: bbr_url_from_name(str(name), seq=1))
    out = df.assign(url=urls)
    return out

if __name__ == "__main__":
    out_df = create_players_urls("a.csv")
    # Save to CSV
    out_df.to_csv("player_urls.csv", index=False, encoding="utf-8")
    print(out_df.head())
