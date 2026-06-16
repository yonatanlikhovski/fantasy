# draft/compute_win_shares.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import numpy as np
import pandas as pd

# ---------- Paths ----------
SIM_RESULTS_CSV   = Path("sim_stats") / "projected_2027_weekly.csv"
FEATURES_CSV      = Path("sim_stats") / "player_features_train_2023_2026.csv"
H2H_VALUE_CSV     = Path("sim_stats") / "h2h_value_2027.csv"

OUT_DIR           = Path("sim_stats")
ROSTERS_OUT       = OUT_DIR / "draft_rosters_2027.csv"
TEAM_AVG_OUT      = OUT_DIR / "draft_team_averages_2027.csv"
STANDINGS_OUT     = OUT_DIR / "draft_standings_2027.csv"
WS_PLAYERS_OUT    = OUT_DIR / "draft_win_shares_players_2027.csv"
WS_SUMMARY_OUT    = OUT_DIR / "draft_win_shares_summary_2027.csv"
WS_VS_H2H_OUT     = OUT_DIR / "ws_vs_h2h_2027.csv"

COUNT_CATS = ["PTS","REB","AST","STL","BLK","FG3M","TOV"]

# ---------- Small helpers ----------
def _z_from_ci_5_95(width: float) -> float:
    # 5–95% interval ≈ 3.29σ → σ = width / 3.29
    return max(0.0, float(width)) / 3.29

def _pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(method="average", pct=True)

def _case_pick(df: pd.DataFrame, *names: str, default=None) -> pd.Series:
    cmap = {c.lower(): c for c in df.columns}
    for n in names:
        k = cmap.get(n.lower())
        if k is not None:
            return df[k]
    if isinstance(default, pd.Series):
        return default
    return pd.Series(default, index=df.index if hasattr(df, "index") else None)

def _clip_nonneg(x: np.ndarray) -> np.ndarray:
    x[x < 0.0] = 0.0
    return x

# ---------- Data loading / pool prep ----------
def load_player_pool(sim_path: Path, feat_path: Path, games_per_week: float) -> pd.DataFrame:
    sr = pd.read_csv(sim_path)
    ft = pd.read_csv(feat_path)
    sr.columns = [c.strip() for c in sr.columns]
    ft.columns = [c.strip() for c in ft.columns]

    base = ft[[c for c in [
        "player_id","durability",
        "FGA_mean","FTA_mean","FGM_mean","FTM_mean","FGP_mean","FTP_mean"
    ] if c in ft.columns]].copy()
    df = sr.merge(base, on="player_id", how="left")

    # per-game attempts (prefer direct; else reconstruct from makes/%)
    FGA_pg = _case_pick(df, "FGA_mean", default=np.nan).astype(float)
    FTA_pg = _case_pick(df, "FTA_mean", default=np.nan).astype(float)
    if FGA_pg.isna().any():
        FGA_pg = FGA_pg.fillna(_case_pick(df, "FGM_mean", default=0).astype(float) /
                               _case_pick(df, "FGP_mean","FGP", default=1).astype(float))
    if FTA_pg.isna().any():
        FTA_pg = FTA_pg.fillna(_case_pick(df, "FTM_mean", default=0).astype(float) /
                               _case_pick(df, "FTP_mean","FTP", default=1).astype(float))

    df["FGA_week"] = FGA_pg.fillna(0).clip(lower=0) * games_per_week
    df["FTA_week"] = FTA_pg.fillna(0).clip(lower=0) * games_per_week
    df["FGP_mean"] = _case_pick(df, "FGP_mean","FGP", default=0.45).astype(float).clip(0,1)
    df["FTP_mean"] = _case_pick(df, "FTP_mean","FTP", default=0.78).astype(float).clip(0,1)

    # σ for counting cats from CI (if missing, zeros)
    for c in COUNT_CATS:
        mean_col = f"{c}_mean"
        lo_col   = f"{c}_ci_lo"
        hi_col   = f"{c}_ci_hi"
        if mean_col not in df.columns:
            df[mean_col] = 0.0
        width = (df[hi_col] - df[lo_col]) if (lo_col in df.columns and hi_col in df.columns) else pd.Series(0.0, index=df.index)
        df[f"{c}_sigma"] = width.fillna(0.0).map(_z_from_ci_5_95)

    df = df.drop_duplicates(subset=["player_id"]).reset_index(drop=True)
    return df

# ---------- Strategy weights ----------
def strategy_weights(name: str) -> Dict[str, float]:
    # weights for per-category draft score (baseline = 1s)
    base = {"PTS":1,"REB":1,"AST":1,"STL":1,"BLK":1,"FG3M":1,"TOV":1,"FG":1,"FT":1}
    n = name.lower()
    if n == "blocks_heavy":
        base.update({"BLK":1.8, "STL":1.2, "PTS":0.8})
    elif n == "guard":
        base.update({"AST":1.5, "FG3M":1.4, "STL":1.2, "REB":0.8, "BLK":0.7})
    elif n == "big":
        base.update({"REB":1.5, "BLK":1.6, "FG":1.3, "FT":0.8, "AST":0.8})
    # add more named strategies if you like
    return base

def apply_punts(weights: Dict[str,float], punts: set[str]) -> Dict[str,float]:
    w = weights.copy()
    for p in punts:
        key = p.upper()
        if key in w:
            w[key] = 0.0
    return w

# ---------- Draft ranking ----------
def compute_draft_ranking(df: pd.DataFrame, risk_aversion: float,
                          weights: Dict[str,float]) -> pd.DataFrame:
    # FG/FT impact vs pool averages (volume-weighted)
    fgp_pool = float(df["FGP_mean"].mean()); ftp_pool = float(df["FTP_mean"].mean())
    df["FG_impact"] = (df["FGP_mean"] - fgp_pool) * df["FGA_week"]
    df["FT_impact"] = (df["FTP_mean"] - ftp_pool) * df["FTA_week"]

    work = pd.DataFrame({"player_id": df["player_id"]})
    for c in ["PTS","REB","AST","STL","BLK","FG3M","TOV"]:
        work[c] = df[f"{c}_mean"].fillna(0.0)
    work["FG"] = df["FG_impact"]; work["FT"] = df["FT_impact"]
    work["TOV_for_rank"] = -work["TOV"]  # invert TOV

    percat = {}
    for c in ["PTS","REB","AST","STL","BLK","FG3M","FG","FT"]:
        percat[c] = 2*_pct_rank(work[c]) - 1
    percat["TOV"] = 2*_pct_rank(work["TOV_for_rank"]) - 1
    scores = pd.DataFrame(percat)

    # light risk penalty using CI widths (counts + TOV)
    risk = np.zeros(len(df))
    for c in ["PTS","REB","AST","STL","BLK","FG3M","TOV"]:
        lo = df.get(f"{c}_ci_lo"); hi = df.get(f"{c}_ci_hi")
        if lo is not None and hi is not None:
            risk += (hi - lo).fillna(0).to_numpy()
    if risk.std() > 0:
        risk = (risk - risk.mean())/risk.std()

    # weighted sum
    cat_list = ["PTS","REB","AST","STL","BLK","FG3M","TOV","FG","FT"]
    wvec = np.array([weights[c] for c in cat_list], dtype=float)
    M = scores[cat_list].to_numpy(float)
    score_raw = (M * wvec).sum(axis=1)
    score = score_raw - risk_aversion * risk

    rank_df = pd.DataFrame({"player_id": df["player_id"], "draft_score": score})
    rank_df = rank_df.sort_values("draft_score", ascending=False).reset_index(drop=True)
    return rank_df

# ---------- Snake draft ----------
def snake_draft(player_rank: pd.DataFrame, num_teams: int, rounds: int, seed: int) -> Dict[int, List[str]]:
    rng = np.random.default_rng(seed)
    order = list(range(num_teams)); rng.shuffle(order)
    rosters = {t: [] for t in range(num_teams)}
    pool = player_rank["player_id"].tolist()
    ptr = 0
    for r in range(rounds):
        picks = order if r % 2 == 0 else list(reversed(order))
        for t in picks:
            if ptr >= len(pool): break
            rosters[t].append(pool[ptr]); ptr += 1
    return rosters

# ---------- Week sampling & matchup ----------
def sample_team_week(df: pd.DataFrame, roster: List[str], seed: int) -> Dict[str, float]:
    sub = df.set_index("player_id").loc[roster]
    rng = np.random.default_rng(seed)
    team = {}
    # counting stats
    for c in ["PTS","REB","AST","STL","BLK","TOV","FG3M"]:
        mu = sub[f"{c}_mean"].to_numpy(float)
        sd = sub[f"{c}_sigma"].to_numpy(float)
        samp = rng.normal(mu, sd)
        team[c] = float(_clip_nonneg(samp).sum())
    # FG/FT via attempts & binomial makes
    A_fg = rng.poisson(lam=np.clip(sub["FGA_week"].to_numpy(float), 0, 300))
    A_ft = rng.poisson(lam=np.clip(sub["FTA_week"].to_numpy(float), 0, 200))
    M_fg = rng.binomial(n=A_fg.astype(int), p=sub["FGP_mean"].to_numpy(float).clip(0,1))
    M_ft = rng.binomial(n=A_ft.astype(int), p=sub["FTP_mean"].to_numpy(float).clip(0,1))
    team["FG_pct"] = float(M_fg.sum()) / float(max(1, A_fg.sum()))
    team["FT_pct"] = float(M_ft.sum()) / float(max(1, A_ft.sum()))
    return team

def play_match(teamA: Dict[str,float], teamB: Dict[str,float]) -> Tuple[float,float,float]:
    a=b=t=0.0
    for c in ["PTS","REB","AST","STL","BLK","FG3M"]:
        if teamA[c] > teamB[c]: a+=1
        elif teamB[c] > teamA[c]: b+=1
        else: t+=1
    if teamA["TOV"] < teamB["TOV"]: a+=1
    elif teamB["TOV"] < teamA["TOV"]: b+=1
    else: t+=1
    for c in ["FG_pct","FT_pct"]:
        if teamA[c] > teamB[c]: a+=1
        elif teamB[c] > teamA[c]: b+=1
        else: t+=1
    return a,b,t

def simulate_rr_with_seeds(df: pd.DataFrame, rosters: Dict[int,List[str]], seeds: np.ndarray) -> pd.DataFrame:
    teams = sorted(rosters.keys()); n = len(teams)
    assert len(seeds) == n*(n-1)//2
    rows=[]; idx=0
    for i in range(n):
        for j in range(i+1, n):
            A=teams[i]; B=teams[j]
            sA, sB = int(seeds[idx]), int(seeds[idx]^0x9E3779B1); idx+=1
            teamA = sample_team_week(df, rosters[A], sA)
            teamB = sample_team_week(df, rosters[B], sB)
            a,b,t = play_match(teamA, teamB)
            rows.append({"team":A,"W":1.0 if a>b else 0.0 if b>a else 0.0,
                         "L":1.0 if b>a else 0.0 if a>b else 0.0,
                         "T":1.0 if a==b else 0.0,
                         "cat_pts":a + 0.5*t})
            rows.append({"team":B,"W":1.0 if b>a else 0.0 if a>b else 0.0,
                         "L":1.0 if a>b else 0.0 if b>a else 0.0,
                         "T":1.0 if a==b else 0.0,
                         "cat_pts":b + 0.5*t})
    return pd.DataFrame(rows)

# ---------- Replacement & WS ----------
def build_replacement(df: pd.DataFrame, undrafted_ids: List[str]) -> pd.Series:
    sub = df[df["player_id"].isin(undrafted_ids)]
    if sub.empty: sub = df.tail(60)
    row = {"player_id":"__REPLACEMENT__",
           "FGA_week":float(sub["FGA_week"].mean()),
           "FTA_week":float(sub["FTA_week"].mean()),
           "FGP_mean":float(sub["FGP_mean"].mean()),
           "FTP_mean":float(sub["FTP_mean"].mean())}
    for c in ["PTS","REB","AST","STL","BLK","TOV","FG3M"]:
        row[f"{c}_mean"]  = float(sub[f"{c}_mean"].mean())
        row[f"{c}_sigma"] = float(sub[f"{c}_sigma"].mean())
    return pd.Series(row)

def compute_ws(df: pd.DataFrame, rosters: Dict[int,List[str]], trials: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    teams = sorted(rosters.keys()); n=len(teams); matches=n*(n-1)//2

    # base seasons
    agg = {"W":0.0,"L":0.0,"T":0.0,"cat_pts":0.0}
    base = {t: agg.copy() for t in teams}
    seeds = rng.integers(1, 2**31-1, size=(trials, matches), dtype=np.int64)
    for tr in range(trials):
        rr = simulate_rr_with_seeds(df, rosters, seeds[tr])
        g = rr.groupby("team")[["W","L","T","cat_pts"]].sum()
        for t in teams:
            base[t]["W"] += float(g.loc[t,"W"])
            base[t]["L"] += float(g.loc[t,"L"])
            base[t]["T"] += float(g.loc[t,"T"])
            base[t]["cat_pts"] += float(g.loc[t,"cat_pts"])
    base_df = pd.DataFrame([{"team":t,
                             "W":v["W"]/trials, "L":v["L"]/trials,
                             "T":v["T"]/trials, "cat_pts":v["cat_pts"]/trials}
                            for t,v in base.items()])

    # replacement profile
    drafted = {p for lst in rosters.values() for p in lst}
    undrafted = [pid for pid in df["player_id"].tolist() if pid not in drafted]
    repl = build_replacement(df, undrafted)
    if "__REPLACEMENT__" not in df["player_id"].values:
        add = pd.DataFrame([repl])
        for col in df.columns:
            if col not in add.columns: add[col]=0.0
        df = pd.concat([df, add[df.columns]], ignore_index=True)

    # per-player WS
    ws_rows=[]
    for team in teams:
        for p in rosters[team]:
            alt = {t:list(lst) for t,lst in rosters.items()}
            try:
                k = alt[team].index(p)
            except ValueError:
                continue
            alt[team][k] = "__REPLACEMENT__"

            alt_w = 0.0
            for tr in range(trials):
                rr = simulate_rr_with_seeds(df, alt, seeds[tr])
                g = rr.groupby("team")["W"].sum()
                alt_w += float(g.loc[team])
            alt_w /= trials
            ws = float(base_df.loc[base_df["team"]==team,"W"].iloc[0]) - alt_w
            ws_rows.append({"team":team,"player_id":p,"win_shares":ws})

    ws_df = pd.DataFrame(ws_rows).sort_values(["team","win_shares"], ascending=[True,False]).reset_index(drop=True)
    return base_df, ws_df

# ---------- Team averages (deterministic, for inspection) ----------
def team_averages(df: pd.DataFrame, rosters: Dict[int,List[str]]) -> pd.DataFrame:
    rows=[]; sub=df.set_index("player_id")
    for t, players in rosters.items():
        P=sub.loc[players]
        row={"team":t}
        for c in ["PTS","REB","AST","STL","BLK","TOV","FG3M"]:
            row[c]=float(P[f"{c}_mean"].sum())
        Mfg=float((P["FGP_mean"]*P["FGA_week"]).sum()); Afg=float(P["FGA_week"].sum())
        Mft=float((P["FTP_mean"]*P["FTA_week"]).sum()); Aft=float(P["FTA_week"].sum())
        row["FG_pct"]=Mfg/Afg if Afg>0 else 0.0
        row["FT_pct"]=Mft/Aft if Aft>0 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

# ---------- H2H comparison (optional) ----------
def try_load_h2h(path: Path) -> pd.DataFrame | None:
    if not path.exists(): return None
    hv = pd.read_csv(path)
    hv.columns = [c.strip() for c in hv.columns]
    for col in ["H2H_value","z_total_avail","VORP","sum_risk_adj","sum_raw","value","score"]:
        if col in hv.columns:
            return hv[["player_id", col]].rename(columns={col:"H2H_value"})
    return None

def compute_ws_fast(df: pd.DataFrame, rosters: Dict[int, List[str]],
                    trials: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    teams = sorted(rosters.keys())
    n = len(teams)
    matches = n * (n - 1) // 2

    # Replacement profile from undrafted pool
    drafted = {p for lst in rosters.values() for p in lst}
    undrafted = [pid for pid in df["player_id"].tolist() if pid not in drafted]
    repl_src = build_replacement(df, undrafted)
    repl = repl_src.to_dict()

    idx = df.set_index("player_id")

    # Aggregates across all trials
    base_wins = {t: 0.0 for t in teams}   # weekly match wins
    base_cat  = {t: 0.0 for t in teams}   # category points (wins + 0.5*ties)
    ws_sum    = {t: {p: 0.0 for p in rosters[t]} for t in teams}

    def sample_player(pid: str, local_rng: np.random.Generator):
        if pid == "__REPLACEMENT__":
            row = repl
        else:
            r = idx.loc[pid]
            row = {
                "PTS_mean":  float(r["PTS_mean"]),  "PTS_sigma":  float(r["PTS_sigma"]),
                "REB_mean":  float(r["REB_mean"]),  "REB_sigma":  float(r["REB_sigma"]),
                "AST_mean":  float(r["AST_mean"]),  "AST_sigma":  float(r["AST_sigma"]),
                "STL_mean":  float(r["STL_mean"]),  "STL_sigma":  float(r["STL_sigma"]),
                "BLK_mean":  float(r["BLK_mean"]),  "BLK_sigma":  float(r["BLK_sigma"]),
                "TOV_mean":  float(r["TOV_mean"]),  "TOV_sigma":  float(r["TOV_sigma"]),
                "FG3M_mean": float(r["FG3M_mean"]), "FG3M_sigma": float(r["FG3M_sigma"]),
                "FGA_week":  float(r["FGA_week"]),  "FGP_mean":   float(r["FGP_mean"]),
                "FTA_week":  float(r["FTA_week"]),  "FTP_mean":   float(r["FTP_mean"]),
            }

        # counting cats ~ Normal(mean, sigma) clipped at 0
        counts = {}
        for c in ["PTS","REB","AST","STL","BLK","TOV","FG3M"]:
            val = float(local_rng.normal(row[f"{c}_mean"], row[f"{c}_sigma"]))
            counts[c] = 0.0 if val < 0.0 else val

        # FG/FT attempts ~ Poisson, makes ~ Binomial
        A_fg = int(local_rng.poisson(lam=min(max(row["FGA_week"], 0.0), 300.0)))
        A_ft = int(local_rng.poisson(lam=min(max(row["FTA_week"], 0.0), 200.0)))
        M_fg = int(local_rng.binomial(n=A_fg, p=np.clip(row["FGP_mean"], 0, 1)))
        M_ft = int(local_rng.binomial(n=A_ft, p=np.clip(row["FTP_mean"], 0, 1)))

        return counts, (M_fg, A_fg, M_ft, A_ft)

    def decide(A, B):
        a = b = t = 0.0
        for c in ["PTS","REB","AST","STL","BLK","FG3M"]:
            if A[c] > B[c]: a += 1
            elif B[c] > A[c]: b += 1
            else: t += 1
        # TOV lower better
        if A["TOV"] < B["TOV"]: a += 1
        elif B["TOV"] < A["TOV"]: b += 1
        else: t += 1
        # FG/FT %
        for pct in ["FG_pct","FT_pct"]:
            if A[pct] > B[pct]: a += 1
            elif B[pct] > A[pct]: b += 1
            else: t += 1
        winA = 1.0 if a > b else 0.0 if b > a else 0.0
        return winA, (a, b, t)

    # --- run trials; sample each match once and reuse for WS deltas ---
    for _ in range(trials):
        match_seeds = rng.integers(1, 2**31 - 1, size=matches, dtype=np.int64)
        mptr = 0
        for i in range(n):
            for j in range(i + 1, n):
                A = teams[i]; B = teams[j]
                rA = np.random.default_rng(int(match_seeds[mptr]))
                rB = np.random.default_rng(int(match_seeds[mptr] ^ 0x9E3779B1))
                mptr += 1

                # sample contributions for both teams
                A_contrib = []; A_tot = {c:0.0 for c in ["PTS","REB","AST","STL","BLK","TOV","FG3M"]}
                A_fgM = A_fgA = A_ftM = A_ftA = 0
                for pid in rosters[A]:
                    counts, (mfg, afg, mft, aft) = sample_player(pid, rA)
                    for c in A_tot: A_tot[c] += counts[c]
                    A_fgM += mfg; A_fgA += afg; A_ftM += mft; A_ftA += aft
                    A_contrib.append((pid, counts, mfg, afg, mft, aft))

                B_contrib = []; B_tot = {c:0.0 for c in ["PTS","REB","AST","STL","BLK","TOV","FG3M"]}
                B_fgM = B_fgA = B_ftM = B_ftA = 0
                for pid in rosters[B]:
                    counts, (mfg, afg, mft, aft) = sample_player(pid, rB)
                    for c in B_tot: B_tot[c] += counts[c]
                    B_fgM += mfg; B_fgA += afg; B_ftM += mft; B_ftA += aft
                    B_contrib.append((pid, counts, mfg, afg, mft, aft))

                # one replacement sample per team-match
                replA_counts, (replA_mfg, replA_afg, replA_mft, replA_aft) = sample_player("__REPLACEMENT__", rA)
                replB_counts, (replB_mfg, replB_afg, replB_mft, replB_aft) = sample_player("__REPLACEMENT__", rB)

                # baseline outcomes
                A_team = {**A_tot, "FG_pct": (A_fgM / max(1, A_fgA)), "FT_pct": (A_ftM / max(1, A_ftA))}
                B_team = {**B_tot, "FG_pct": (B_fgM / max(1, B_fgA)), "FT_pct": (B_ftM / max(1, B_ftA))}
                A_win, (a, b, t) = decide(A_team, B_team)
                B_win, _         = decide(B_team, A_team)

                base_wins[A] += A_win
                base_wins[B] += B_win
                base_cat[A]  += a + 0.5 * t
                base_cat[B]  += b + 0.5 * t

                # WS for players on A: swap with replacement (no resampling)
                for pid, counts, mfg, afg, mft, aft in A_contrib:
                    A_alt = {c: A_tot[c] - counts[c] + replA_counts[c] for c in A_tot.keys()}
                    fgM = A_fgM - mfg + replA_mfg; fgA = A_fgA - afg + replA_afg
                    ftM = A_ftM - mft + replA_mft; ftA = A_ftA - aft + replA_aft
                    A_alt["FG_pct"] = fgM / max(1, fgA)
                    A_alt["FT_pct"] = ftM / max(1, ftA)
                    alt_win, _ = decide(A_alt, B_team)
                    ws_sum[A][pid] += (A_win - alt_win)

                # WS for players on B
                for pid, counts, mfg, afg, mft, aft in B_contrib:
                    B_alt = {c: B_tot[c] - counts[c] + replB_counts[c] for c in B_tot.keys()}
                    fgM = B_fgM - mfg + replB_mfg; fgA = B_fgA - afg + replB_afg
                    ftM = B_ftM - mft + replB_mft; ftA = B_ftA - aft + replB_aft
                    B_alt["FG_pct"] = fgM / max(1, fgA)
                    B_alt["FT_pct"] = ftM / max(1, ftA)
                    alt_win, _ = decide(B_alt, A_team)
                    ws_sum[B][pid] += (B_win - alt_win)

    # standings with W and cat_pts (averaged over trials)
    standings = pd.DataFrame([
        {"team": t, "W": base_wins[t] / trials, "cat_pts": base_cat[t] / trials}
        for t in teams
    ]).sort_values(["W","cat_pts"], ascending=[False, False]).reset_index(drop=True)

    # per-player WS (average per trial)
    ws_rows = []
    for t in teams:
        for p in rosters[t]:
            ws_rows.append({"team": t, "player_id": p, "win_shares": ws_sum[t][p] / trials})
    ws_df = pd.DataFrame(ws_rows).sort_values(["team","win_shares"], ascending=[True, False]).reset_index(drop=True)
    return standings, ws_df

# ---------------------------------------------------------------------------

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Draft sim + per-player Win Shares; compare to H2H_value if present.")
    ap.add_argument("--teams", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=13)
    ap.add_argument("--trials", type=int, default=250)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--gpw", type=float, default=3.5, help="games per week used to scale attempts")
    ap.add_argument("--risk", type=float, default=0.10, help="risk penalty strength for draft ranking")
    ap.add_argument("--strategy", type=str, default="baseline", help="baseline | blocks_heavy | guard | big")
    ap.add_argument("--punt", type=str, default="", help="comma list of cats to punt (e.g. FT,TOV)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load pool
    df = load_player_pool(SIM_RESULTS_CSV, FEATURES_CSV, args.gpw)

    # Strategy weights (+ punts)
    w = strategy_weights(args.strategy)
    punts = set([x.strip().upper() for x in args.punt.split(",") if x.strip()])
    w = apply_punts(w, punts)

    # Rank & draft
    rank = compute_draft_ranking(df, risk_aversion=args.risk, weights=w)
    rosters = snake_draft(rank, num_teams=args.teams, rounds=args.rounds, seed=args.seed)
    pd.DataFrame([(t,p) for t, lst in rosters.items() for p in lst], columns=["team","player_id"]).to_csv(ROSTERS_OUT, index=False)

    # Team averages (deterministic)
    team_avg = team_averages(df, rosters); team_avg.to_csv(TEAM_AVG_OUT, index=False)

    # Win Shares via simulation
    standings, ws_players = compute_ws_fast(df, rosters, trials=args.trials, seed=args.seed)
    standings = standings.sort_values(["W","cat_pts"], ascending=[False,False]).reset_index(drop=True)
    standings.to_csv(STANDINGS_OUT, index=False)
    ws_players.to_csv(WS_PLAYERS_OUT, index=False)
    ws_players.groupby("team", as_index=False)["win_shares"].sum().rename(
        columns={"win_shares":"team_win_shares_sum"}
    ).to_csv(WS_SUMMARY_OUT, index=False)

    print("Saved:")
    print(f" - {ROSTERS_OUT}")
    print(f" - {TEAM_AVG_OUT}")
    print(f" - {STANDINGS_OUT}")
    print(f" - {WS_PLAYERS_OUT}")
    print(f" - {WS_SUMMARY_OUT}")

    # Optional: compare WS to H2H_value
    hv = try_load_h2h(H2H_VALUE_CSV)
    if hv is not None:
        comp = ws_players.merge(hv, on="player_id", how="left")
        # Rank by each metric (higher better)
        comp["rank_WS"]  = comp["win_shares"].rank(ascending=False, method="min")
        comp["rank_H2H"] = comp["H2H_value"].rank(ascending=False, method="min")
        comp["rank_diff"] = comp["rank_WS"] - comp["rank_H2H"]
        comp.to_csv(WS_VS_H2H_OUT, index=False)
        s = comp[["win_shares","H2H_value"]].dropna()
        if len(s) >= 3:
            pear = float(s.corr(method="pearson").iloc[0,1])
            spear = float(s.corr(method="spearman").iloc[0,1])
            print(f"\nWS vs H2H_value correlation: Pearson={pear:.3f}, Spearman={spear:.3f}, n={len(s)}")
        print(f" - {WS_VS_H2H_OUT}")
    else:
        print("\nNo h2h_value.csv found — skipped WS↔H2H comparison.")

if __name__ == "__main__":
    main()
