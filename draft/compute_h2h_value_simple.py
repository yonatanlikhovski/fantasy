# compute_h2h_value_simple.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COUNT_CATS = ["PTS", "REB", "AST", "STL", "BLK", "FG3M"]
NEG_CAT = "TOV"


def pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(method="average", pct=True)


def zscore(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=1)

    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)

    return (s - mu) / sd


def pick(df: pd.DataFrame, *cands: str, default=None) -> pd.Series:
    """
    Case-insensitive column getter.
    Returns a default Series if no candidate exists.
    """
    cmap = {c.lower(): c for c in df.columns}

    for name in cands:
        key = cmap.get(name.lower())
        if key is not None:
            return pd.to_numeric(df[key], errors="coerce")

    if isinstance(default, pd.Series):
        return default

    return pd.Series(default, index=df.index)


def ci_width(df: pd.DataFrame, base: str) -> pd.Series:
    lo = df.get(f"{base}_ci_lo")
    hi = df.get(f"{base}_ci_hi")

    if lo is None or hi is None:
        return pd.Series(0.0, index=df.index)

    return (
        pd.to_numeric(hi, errors="coerce")
        - pd.to_numeric(lo, errors="coerce")
    ).fillna(0.0)


def compute_h2h_value(
    weekly_df: pd.DataFrame,
    risk_aversion: float = 0.10,
    use_availability_multiplier: bool = False,
) -> pd.DataFrame:
    """
    Compute fantasy H2H value from weekly fantasy stats.

    Input should contain:
        player_id
        PTS_mean, REB_mean, AST_mean, STL_mean, BLK_mean, FG3M_mean, TOV_mean
        FG%_mean or FGP_mean
        FT%_mean or FTP_mean
        FGA_mean
        FTA_mean

    Important:
        By default, use_availability_multiplier=False because projected weekly
        stats already include games missed through the Monte Carlo simulation.
    """
    df = weekly_df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "player_id" not in df.columns:
        raise KeyError("Input weekly CSV must contain a 'player_id' column.")

    out_base = pd.DataFrame({"player_id": df["player_id"]})
    work = pd.DataFrame({"player_id": df["player_id"]})

    # Counting categories
    for cat in COUNT_CATS:
        work[cat] = pick(df, f"{cat}_mean", cat, default=0.0).fillna(0.0)

    # Turnovers: lower is better
    work[NEG_CAT] = pick(df, f"{NEG_CAT}_mean", NEG_CAT, default=0.0).fillna(0.0)
    work["TOV_for_rank"] = -work[NEG_CAT]

    # Percent categories
    fgp = pick(df, "FG%_mean", "FGP_mean", "FG%", "FGP", default=np.nan)
    ftp = pick(df, "FT%_mean", "FTP_mean", "FT%", "FTP", default=np.nan)

    fga = pick(df, "FGA_mean", "FGA", default=0.0).fillna(0.0)
    fta = pick(df, "FTA_mean", "FTA", default=0.0).fillna(0.0)

    fgp = fgp.fillna(fgp.mean() if not np.isnan(fgp.mean()) else 0.45)
    ftp = ftp.fillna(ftp.mean() if not np.isnan(ftp.mean()) else 0.78)

    # Volume-weighted percentage impact
    work["FG"] = (fgp - fgp.mean()) * fga
    work["FT"] = (ftp - ftp.mean()) * fta

    # Percentile score per category: [-1, +1]
    scores = pd.DataFrame(index=df.index)

    for cat in COUNT_CATS:
        scores[cat] = 2 * pct_rank(work[cat]) - 1

    scores["TOV"] = 2 * pct_rank(work["TOV_for_rank"]) - 1
    scores["FG"] = 2 * pct_rank(work["FG"]) - 1
    scores["FT"] = 2 * pct_rank(work["FT"]) - 1

    cat_cols = ["PTS", "REB", "AST", "STL", "BLK", "FG3M", "TOV", "FG", "FT"]

    # Risk penalty from confidence interval width, if CI columns exist.
    risk = np.zeros(len(df))

    for cat in COUNT_CATS + [NEG_CAT]:
        risk += ci_width(df, cat).to_numpy()

    risk = zscore(pd.Series(risk, index=df.index)).fillna(0.0).to_numpy()

    scores["sum_raw"] = scores[cat_cols].sum(axis=1)
    scores["risk_penalty"] = risk_aversion * risk
    scores["H2H_value"] = scores["sum_raw"] - scores["risk_penalty"]

    if use_availability_multiplier:
        durability = pick(df, "durability", "games_played_mean", default=82).fillna(82)
        availability_multiplier = durability.clip(0, 82) / 82.0
        scores["availability_multiplier"] = availability_multiplier
        scores["H2H_value"] = scores["H2H_value"] * availability_multiplier
    else:
        scores["availability_multiplier"] = 1.0

    scores["rank_H2H"] = scores["H2H_value"].rank(
        ascending=False,
        method="min"
    )

    out = pd.concat(
        [
            out_base,
            pd.DataFrame({
                "FGA_mean": fga,
                "FTA_mean": fta,
                "FG_pct_used": fgp,
                "FT_pct_used": ftp,
            }),
            scores,
        ],
        axis=1,
    )

    out = out.sort_values("H2H_value", ascending=False).reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Compute fantasy H2H value from a weekly stats CSV."
    )

    parser.add_argument(
        "--weekly",
        type=Path,
        required=True,
        help="Input weekly stats CSV, e.g. sim_stats/projected_2026_weekly.csv",
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output H2H value CSV.",
    )

    parser.add_argument(
        "--risk",
        type=float,
        default=0.10,
        help="Risk penalty strength. Default: 0.10",
    )

    parser.add_argument(
        "--availability-mult",
        action="store_true",
        help="Multiply by availability. Usually leave OFF for simulated weekly outputs.",
    )

    args = parser.parse_args()

    weekly = pd.read_csv(args.weekly)
    out = compute_h2h_value(
        weekly,
        risk_aversion=args.risk,
        use_availability_multiplier=args.availability_mult,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"✅ H2H values written -> {args.out}")
    print(out[["player_id", "H2H_value", "rank_H2H"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()