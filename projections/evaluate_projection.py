from pathlib import Path
import numpy as np
import pandas as pd

STATS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FG%", "FT%", "games_played"]


def corr(x, y, method):
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3:
        return np.nan
    return frame["x"].corr(frame["y"], method=method)


def main():
    pred_path = Path("sim_stats") / "projected_2026_weekly.csv"
    actual_path = Path("sim_stats") / "actual_2026_weekly.csv"

    pred = pd.read_csv(pred_path)
    actual = pd.read_csv(actual_path)

    # Debug/exclusion lists before renaming columns
    pred_ids = set(pred["player_id"])
    actual_ids = set(actual["player_id"])

    actual_only = actual[~actual["player_id"].isin(pred_ids)].copy()
    projected_only = pred[~pred["player_id"].isin(actual_ids)].copy()

    actual_only_path = Path("sim_stats") / "actual_only_2026_players.csv"
    projected_only_path = Path("sim_stats") / "projected_only_2026_players.csv"

    actual_only.to_csv(actual_only_path, index=False)
    projected_only.to_csv(projected_only_path, index=False)

    print(f"Actual 2026 players: {len(actual)}")
    print(f"Projected players: {len(pred)}")
    print(f"Overlap / evaluated players: {len(actual_ids & pred_ids)}")
    print(f"Actual-only / no-history players: {len(actual_only)} -> {actual_only_path}")
    print(f"Projected-only players: {len(projected_only)} -> {projected_only_path}")

    # Rename after saving exclusion files
    pred = pred.rename(columns={c: f"pred_{c}" for c in pred.columns if c != "player_id"})
    actual = actual.rename(columns={c: f"actual_{c}" for c in actual.columns if c != "player_id"})

    # Evaluate only players that exist in both train projection and actual 2026
    df = pred.merge(
        actual,
        on="player_id",
        how="inner",
        validate="one_to_one"
    )

    # Suspicious cases: likely wrong ID collision, not normal model error
    suspicious = df[
        (df.get("pred_games_played_mean", 0) < 25)
        & (df.get("actual_games_played_mean", 0) > 50)
        & (df.get("actual_PTS_mean", 0) > 30)
    ].copy()

    suspicious_path = Path("sim_stats") / "suspicious_id_matches.csv"
    suspicious.to_csv(suspicious_path, index=False)

    if not suspicious.empty:
        print(f"\n⚠️ Suspicious ID matches written to {suspicious_path}")
        print(
            suspicious[
                [
                    "player_id",
                    "pred_games_played_mean",
                    "actual_games_played_mean",
                    "pred_PTS_mean",
                    "actual_PTS_mean",
                ]
            ].to_string(index=False)
        )

    summary_rows = []

    for stat in STATS:
        pred_col = f"pred_{stat}_mean"
        actual_col = f"actual_{stat}_mean"

        if pred_col not in df.columns or actual_col not in df.columns:
            continue

        err_col = f"{stat}_error"
        abs_col = f"{stat}_abs_error"

        df[err_col] = (
            pd.to_numeric(df[pred_col], errors="coerce")
            - pd.to_numeric(df[actual_col], errors="coerce")
        )
        df[abs_col] = df[err_col].abs()

        valid = df[[pred_col, actual_col, err_col, abs_col]].dropna()

        if valid.empty:
            continue

        ci_lo = f"pred_{stat}_ci_lo"
        ci_hi = f"pred_{stat}_ci_hi"

        if ci_lo in df.columns and ci_hi in df.columns:
            coverage = ((df[actual_col] >= df[ci_lo]) & (df[actual_col] <= df[ci_hi])).mean()
        else:
            coverage = np.nan

        summary_rows.append({
            "stat": stat,
            "n": len(valid),
            "mae": valid[abs_col].mean(),
            "rmse": np.sqrt((valid[err_col] ** 2).mean()),
            "bias_pred_minus_actual": valid[err_col].mean(),
            "pearson": corr(df[pred_col], df[actual_col], "pearson"),
            "spearman": corr(df[pred_col], df[actual_col], "spearman"),
            "ci_5_95_coverage": coverage,
        })

    out = Path("sim_stats") / "evaluation_2026.csv"
    summary_out = Path("sim_stats") / "evaluation_2026_summary.csv"

    df.to_csv(out, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_out, index=False)

    print(f"\n✅ Wrote player-level evaluation -> {out}")
    print(f"✅ Wrote summary -> {summary_out}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()