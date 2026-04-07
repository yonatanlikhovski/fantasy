# draft/merge_rankings.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Defaults (change if your paths differ)
H2H_PATH_DEFAULT        = Path("sim_stats") / "h2h_value.csv"
WS_DRAFT_PATH_DEFAULT   = Path("sim_stats") / "draft_win_shares_players.csv"
WS_STRAT_PATH_DEFAULT   = Path("sim_stats") / "strategic_draft_win_shares_players.csv"
OUT_PATH_DEFAULT        = Path("sim_stats") / "all_rankings_combined.csv"

def _normcols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cmap = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cmap:
            return cmap[name.lower()]
    return None

def load_h2h(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = _normcols(pd.read_csv(path))
    pid = _pick_col(df, ["player_id","Player","Name"])
    if pid is None:
        return None
    val = _pick_col(df, ["H2H_value","z_total_avail","VORP","sum_risk_adj","sum_raw","value","score"])
    if val is None:
        return None
    out = df[[pid, val]].rename(columns={pid:"player_id", val:"H2H_value"}).copy()
    out["rank_h2h"] = out["H2H_value"].rank(ascending=False, method="min")
    return out

def load_ws(path: Path, *, label_prefix: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = _normcols(pd.read_csv(path))
    pid = _pick_col(df, ["player_id","Player","Name"])
    if pid is None:
        return None
    wscol = _pick_col(df, ["win_shares","WS","ws"])
    if wscol is None:
        return None
    keep = [pid, wscol]
    # Try to preserve team/strategy info if present
    team = _pick_col(df, ["team","Team"])
    if team: keep.append(team)
    strat = _pick_col(df, ["strategy"])
    if strat: keep.append(strat)
    punts = _pick_col(df, ["punts"])
    if punts: keep.append(punts)
    risk  = _pick_col(df, ["risk"])
    if risk: keep.append(risk)

    out = df[keep].rename(columns={
        pid: "player_id",
        wscol: f"{label_prefix}_WS",
        **({team: f"{label_prefix}_team"} if team else {}),
        **({strat: f"{label_prefix}_strategy"} if strat else {}),
        **({punts: f"{label_prefix}_punts"} if punts else {}),
        **({risk: f"{label_prefix}_risk"} if risk else {}),
    })
    out[f"rank_{label_prefix}_WS"] = out[f"{label_prefix}_WS"].rank(ascending=False, method="min")
    return out

def main():
    ap = argparse.ArgumentParser(description="Merge H2H, draft WS, and strategic-draft WS into one CSV")
    ap.add_argument("--h2h", type=Path, default=H2H_PATH_DEFAULT)
    ap.add_argument("--ws_draft", type=Path, default=WS_DRAFT_PATH_DEFAULT)
    ap.add_argument("--ws_strat", type=Path, default=WS_STRAT_PATH_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_PATH_DEFAULT)
    args = ap.parse_args()

    parts = []
    h2h = load_h2h(args.h2h)
    if h2h is not None:
        parts.append(h2h)
    ws_draft = load_ws(args.ws_draft, label_prefix="draft")
    if ws_draft is not None:
        parts.append(ws_draft)
    ws_strat = load_ws(args.ws_strat, label_prefix="strat")
    if ws_strat is not None:
        parts.append(ws_strat)

    if not parts:
        raise SystemExit("No input tables found. Check your paths or generate the CSVs first.")

    # Outer-join on player_id so nobody gets dropped
    merged = parts[0]
    for p in parts[1:]:
        merged = merged.merge(p, on="player_id", how="outer")

    # Optional: consensus rank (average of available ranks)
    rank_cols = [c for c in merged.columns if c.startswith("rank_")]
    if rank_cols:
        merged["consensus_rank_mean"] = merged[rank_cols].mean(axis=1, skipna=True)
        merged["consensus_rank_median"] = merged[rank_cols].median(axis=1, skipna=True)

    # Sort by best available signal: prefer WS_strat → WS_draft → H2H
    sort_cols = [c for c in ["strat_WS","draft_WS","H2H_value"] if c in merged.columns]
    merged = merged.sort_values(sort_cols, ascending=[False]*len(sort_cols)).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"✅ Wrote combined rankings → {args.out}")
    show = ["player_id","H2H_value","rank_h2h","draft_WS","rank_draft_WS","strat_WS","rank_strat_WS","consensus_rank_mean"]
    show = [c for c in show if c in merged.columns]
    print(merged[show].head(20).to_string(index=False))

if __name__ == "__main__":
    main()
