# draft/compute_h2h_value.py
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- Paths ----------
FEATURES_CSV = Path("sim_stats") / "player_features_summary.csv"
SIM_CSV      = Path("sim_stats") / "weekly_sim_results.csv"
OUT_CSV      = Path("sim_stats") / "h2h_value.csv"

# ---------- Config ----------
GAMES_PER_WEEK = 3.5     # same value you used in the simulator
RISK_AVERSION  = 0.10    # 0..0.3 reasonable; higher = more conservative

# Cats
COUNT_CATS = ["PTS","REB","AST","STL","BLK","FG3M"]  # from *_mean in weekly_sim_results
NEG_CAT    = "TOV"                                   # lower is better
PCT_COLS   = {"FG": "FGP_mean", "FT": "FTP_mean"}    # percent categories in weekly_sim_results

# ---------- Helpers ----------
def pct_rank(s: pd.Series) -> pd.Series:
    # percentile in [0,1]; average ties handling
    return s.rank(method="average", pct=True)

def zscore(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

def pick(df: pd.DataFrame, *cands: str, default=None) -> pd.Series:
    """Case-insensitive first-match column getter; returns default (Series or scalar) if none."""
    cmap = {c.lower(): c for c in df.columns}
    for name in cands:
        key = cmap.get(name.lower())
        if key is not None:
            return df[key]
    if isinstance(default, pd.Series):
        return default
    return pd.Series(default, index=df.index)

def ci_width(df: pd.DataFrame, base: str) -> pd.Series:
    lo = df.get(f"{base}_ci_lo")
    hi = df.get(f"{base}_ci_hi")
    if lo is None or hi is None:
        return pd.Series(0.0, index=df.index)
    return (hi - lo).fillna(0)

# ---------- Main ----------
def main():
    # Load
    f = pd.read_csv(FEATURES_CSV)
    r = pd.read_csv(SIM_CSV)

    # Normalize names (trim only; keep original case for readability)
    f.columns = [c.strip() for c in f.columns]
    r.columns = [c.strip() for c in r.columns]

    # Bring only what we need from features (durability + FGA/FTA + optional FGM/FTM/FGP/FTP for reconstruction)
    feat_cols = [c for c in [
        "player_id", "durability",
        "FGA_mean", "FTA_mean",
        "FGM_mean", "FTM_mean",
        "FGP_mean", "FTP_mean",
    ] if c in f.columns]
    base = f[feat_cols].copy()

    # Merge
    df = r.merge(base, on="player_id", how="left")

    # Availability multiplier
    dur = pick(df, "durability", default=74).fillna(74).astype(float).clip(0, 82)
    df["avail_mult"] = dur / 82.0

    # --- Get or reconstruct per-game attempt volumes from features ---
    # Prefer direct FGA_mean / FTA_mean (per-game). If missing, reconstruct from makes/%.
    FGA_pg = pick(df, "FGA_mean", default=np.nan)
    FTA_pg = pick(df, "FTA_mean", default=np.nan)

    if FGA_pg.isna().any():
        FGM_pg = pick(df, "FGM_mean", default=np.nan).astype(float)
        FGP    = pick(df, "FGP_mean", default=np.nan).astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            FGA_pg = FGA_pg.fillna(FGM_pg / FGP)
    if FTA_pg.isna().any():
        FTM_pg = pick(df, "FTM_mean", default=np.nan).astype(float)
        FTP    = pick(df, "FTP_mean", default=np.nan).astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            FTA_pg = FTA_pg.fillna(FTM_pg / FTP)

    FGA_pg = FGA_pg.fillna(0.0).astype(float)
    FTA_pg = FTA_pg.fillna(0.0).astype(float)

    # Weekly attempts (for impact weighting)
    df["FGA_week"] = FGA_pg * GAMES_PER_WEEK
    df["FTA_week"] = FTA_pg * GAMES_PER_WEEK

    # Pool averages for FG%/FT% (from weekly sim results -> PCT_COLS)
    FGP_mean = pick(df, PCT_COLS["FG"], default=0.45).astype(float).fillna(0.45)
    FTP_mean = pick(df, PCT_COLS["FT"], default=0.78).astype(float).fillna(0.78)
    fgp_pool = float(FGP_mean.mean())
    ftp_pool = float(FTP_mean.mean())

    # Impact-style FG / FT (volume-weighted deviation from pool avg)
    df["FG_impact"] = (FGP_mean - fgp_pool) * df["FGA_week"]
    df["FT_impact"] = (FTP_mean - ftp_pool) * df["FTA_week"]

    # Build a working table of category values
    work = pd.DataFrame({"player_id": df["player_id"]})
    for c in COUNT_CATS + [NEG_CAT]:
        col = f"{c}_mean"
        work[c] = df[col] if col in df.columns else 0.0
    work["FG"] = df["FG_impact"]
    work["FT"] = df["FT_impact"]

    # Invert TOV (lower better) for ranking
    work["TOV_for_rank"] = -work["TOV"]

    # Percentile → [-1, +1] per category
    percat = {}
    for c in COUNT_CATS + ["FG","FT"]:
        percat[c] = 2 * pct_rank(work[c]) - 1
    percat["TOV"] = 2 * pct_rank(work["TOV_for_rank"]) - 1
    scores = pd.DataFrame(percat)

    # Risk penalty from CI widths (counts + TOV; FG/FT CIs usually absent and noisy)
    risk = np.zeros(len(df))
    for c in COUNT_CATS + [NEG_CAT]:
        risk += ci_width(df, c).to_numpy()
    risk = zscore(pd.Series(risk)).fillna(0).to_numpy()

    scores["sum_raw"] = scores[["PTS","REB","AST","STL","BLK","FG3M","TOV","FG","FT"]].sum(axis=1)
    scores["sum_risk_adj"] = scores["sum_raw"] - RISK_AVERSION * risk
    scores["H2H_value"] = scores["sum_risk_adj"] * df["avail_mult"]

    out = pd.concat([
        df[["player_id","avail_mult","FGA_week","FTA_week"]],
        scores
    ], axis=1).sort_values("H2H_value", ascending=False).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"✅ H2H values written → {OUT_CSV}")
    print(out.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
