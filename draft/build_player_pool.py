# draft/build_player_pool.py
from pathlib import Path
import pandas as pd
import numpy as np

CATS = ["FG%", "FT%", "FG3M", "PTS", "REB", "AST", "STL", "BLK", "TOV"]

def zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd

def build_player_pool(
    sim_csv: Path = Path("sim_stats/weekly_sim_results.csv"),
    out_csv: Path = Path("draft/player_pool_ranked.csv")
) -> pd.DataFrame:
    if not sim_csv.exists():
        raise FileNotFoundError(f"Could not find {sim_csv.resolve()}")

    df = pd.read_csv(sim_csv)

    needed = ["player_id"] + [f"{c}_mean" for c in CATS]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {sim_csv}: {missing}")

    pool = df[needed].copy()

    # per-category z, flip TOV so lower is better
    for c in CATS:
        z = zscore(pool[f"{c}_mean"])
        pool[f"z_{c}"] = -z if c == "TOV" else z

    # balanced score = sum of z across all cats
    pool["draft_score_balanced"] = pool[[f"z_{c}" for c in CATS]].sum(axis=1)

    # sort best → worst
    cols_out = (
        ["player_id", "draft_score_balanced"]
        + [f"{c}_mean" for c in CATS]
        + [f"z_{c}" for c in CATS]
    )
    pool = pool.sort_values("draft_score_balanced", ascending=False).reset_index(drop=True)
    pool[cols_out].to_csv(out_csv, index=False)

    print(f"Saved ranked pool → {out_csv.resolve()}")
    return pool[cols_out]

if __name__ == "__main__":
    build_player_pool()


