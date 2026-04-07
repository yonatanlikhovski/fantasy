# draft/compare_value_to_h2h.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# -------- Paths --------
SIM_RESULTS_CSV = Path("sim_stats") / "weekly_sim_results.csv"
FEATURES_CSV    = Path("sim_stats") / "player_features_summary.csv"
H2H_VALUE_CSV   = Path("sim_stats") / "h2h_value.csv"
OUT_CSV         = Path("sim_stats") / "value_compare_h2h.csv"

# -------- Config --------
GAMES_PER_WEEK = 3.5    # same GPW used in your simulator
RISK_AVERSION  = 0.10   # 0..0.3 reasonable

COUNT_CATS = ["PTS","REB","AST","STL","BLK","FG3M"]
NEG_CAT    = "TOV"      # lower is better
PCT_COLS   = {"FG": "FGP_mean", "FT": "FTP_mean"}  # names in weekly_sim_results

def pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(method="average", pct=True)

def zscore(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

def pick(df: pd.DataFrame, *names: str, default=None) -> pd.Series:
    cmap = {c.lower(): c for c in df.columns}
    for n in names:
        k = cmap.get(n.lower())
        if k is not None:
            return df[k]
    if isinstance(default, pd.Series):
        return default
    return pd.Series(default, index=df.index)

def ci_width(df: pd.DataFrame, base: str) -> pd.Series:
    lo = df.get(f"{base}_ci_lo")
    hi = df.get(f"{base}_ci_hi")
    if lo is None or hi is None:
        return pd.Series(0.0, index=df.index)
    return (hi - lo).fillna(0)

def load_pool_for_draft_value() -> pd.DataFrame:
    """Merge sim results + features; compute weekly attempts and FG/FT impact."""
    sr = pd.read_csv(SIM_RESULTS_CSV); sr.columns = [c.strip() for c in sr.columns]
    ft = pd.read_csv(FEATURES_CSV);    ft.columns = [c.strip() for c in ft.columns]

    base = ft[[c for c in [
        "player_id","durability",
        "FGA_mean","FTA_mean","FGM_mean","FTM_mean","FGP_mean","FTP_mean"
    ] if c in ft.columns]].copy()

    df = sr.merge(base, on="player_id", how="left")

    # per-game attempts (prefer direct; reconstruct if missing)
    FGA_pg = pick(df, "FGA_mean", default=np.nan).astype(float)
    FTA_pg = pick(df, "FTA_mean", default=np.nan).astype(float)
    if FGA_pg.isna().any():
        FGA_pg = FGA_pg.fillna(pick(df, "FGM_mean", default=0).astype(float) /
                               pick(df, "FGP_mean","FGP", default=1).astype(float))
    if FTA_pg.isna().any():
        FTA_pg = FTA_pg.fillna(pick(df, "FTM_mean", default=0).astype(float) /
                               pick(df, "FTP_mean","FTP", default=1).astype(float))

    df["FGA_week"] = FGA_pg.fillna(0).clip(lower=0)*GAMES_PER_WEEK
    df["FTA_week"] = FTA_pg.fillna(0).clip(lower=0)*GAMES_PER_WEEK

    # FG/FT pct from sim results
    df["FGP_mean"] = pick(df, "FGP_mean","FGP", default=0.45).astype(float).clip(0,1)
    df["FTP_mean"] = pick(df, "FTP_mean","FTP", default=0.78).astype(float).clip(0,1)

    # ensure counting means exist
    for c in COUNT_CATS + [NEG_CAT]:
        col = f"{c}_mean"
        if col not in df.columns:
            df[col] = 0.0

    return df

def compute_draft_value(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile-based swing score with volume-weighted FG/FT impact and a light risk penalty."""
    # volume-weighted impact vs pool average
    fgp_pool = float(df["FGP_mean"].mean())
    ftp_pool = float(df["FTP_mean"].mean())
    df["FG_impact"] = (df["FGP_mean"] - fgp_pool) * df["FGA_week"]
    df["FT_impact"] = (df["FTP_mean"] - ftp_pool) * df["FTA_week"]

    work = pd.DataFrame({"player_id": df["player_id"]})
    for c in COUNT_CATS + [NEG_CAT]:
        work[c] = df[f"{c}_mean"].fillna(0.0)
    work["FG"] = df["FG_impact"]
    work["FT"] = df["FT_impact"]
    work["TOV_for_rank"] = -work["TOV"]  # invert TOV

    # per-cat percentile → [-1,+1]
    percat = {}
    for c in COUNT_CATS + ["FG","FT"]:
        percat[c] = 2*pct_rank(work[c]) - 1
    percat["TOV"] = 2*pct_rank(work["TOV_for_rank"]) - 1
    scores = pd.DataFrame(percat)

    # risk penalty from CI widths across counting cats (if present)
    risk = np.zeros(len(df))
    for c in COUNT_CATS + [NEG_CAT]:
        risk += ci_width(df, c).to_numpy()
    risk = zscore(pd.Series(risk)).fillna(0).to_numpy()

    scores["draft_value_raw"]  = scores[["PTS","REB","AST","STL","BLK","FG3M","TOV","FG","FT"]].sum(axis=1)
    scores["draft_value"]      = scores["draft_value_raw"] - RISK_AVERSION * risk

    return pd.concat([df[["player_id","FGA_week","FTA_week","FGP_mean","FTP_mean"]],
                      scores[["draft_value","draft_value_raw"]]], axis=1)

def load_h2h_value() -> pd.DataFrame:
    hv = pd.read_csv(H2H_VALUE_CSV); hv.columns = [c.strip() for c in hv.columns]
    # find the right column in case you renamed later
    for col in ["H2H_value","z_total_avail","VORP","sum_risk_adj","sum_raw","value","score"]:
        if col in hv.columns:
            return hv[["player_id", col]].rename(columns={col: "H2H_value"})
    raise ValueError("Could not find an H2H value column in h2h_value.csv")

def main():
    df = load_pool_for_draft_value()
    draft = compute_draft_value(df)
    h2h = load_h2h_value()

    merged = draft.merge(h2h, on="player_id", how="left")

    # ranks & diffs
    merged["rank_draft"] = merged["draft_value"].rank(ascending=False, method="min")
    merged["rank_h2h"]   = merged["H2H_value"].rank(ascending=False, method="min")
    merged["value_diff"] = merged["draft_value"] - merged["H2H_value"]
    merged["rank_diff"]  = merged["rank_draft"] - merged["rank_h2h"]

    # quick correlations
    s = merged[["draft_value","H2H_value"]].dropna()
    pear = float(s.corr(method="pearson").iloc[0,1]) if len(s)>=3 else np.nan
    spear = float(s.corr(method="spearman").iloc[0,1]) if len(s)>=3 else np.nan

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.sort_values("draft_value", ascending=False).to_csv(OUT_CSV, index=False)

    print(f"✅ Wrote comparison → {OUT_CSV}")
    if not np.isnan(pear):
        print(f"Correlation(draft_value vs H2H_value): Pearson={pear:.3f}, Spearman={spear:.3f}")
    print("\nTop 15 by draft_value:")
    print(merged.sort_values("draft_value", ascending=False)[
        ["player_id","draft_value","H2H_value","rank_draft","rank_h2h","rank_diff","value_diff"]
    ].head(15).to_string(index=False))

    print("\nBiggest disagreements (by |rank_diff|):")
    disagree = merged.dropna(subset=["rank_diff"]).copy()
    disagree["abs_rank_diff"] = disagree["rank_diff"].abs()
    print(disagree.sort_values("abs_rank_diff", ascending=False)[
        ["player_id","draft_value","H2H_value","rank_draft","rank_h2h","rank_diff","value_diff"]
    ].head(15).to_string(index=False))

if __name__ == "__main__":
    main()
