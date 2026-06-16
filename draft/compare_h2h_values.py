# compare_h2h_values.py
from pathlib import Path
import numpy as np
import pandas as pd


PRED_H2H = Path("sim_stats") / "projected_2026_h2h_value.csv"
ACTUAL_H2H = Path("sim_stats") / "actual_2026_h2h_value.csv"

OUT_DIR = Path("sim_stats") / "evaluations" / "h2h_2026"
OUT_PLAYER = OUT_DIR / "h2h_value_eval_2026.csv"
OUT_SUMMARY = OUT_DIR / "h2h_value_eval_2026_summary.csv"


def corr(x, y, method: str):
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3:
        return np.nan
    return frame["x"].corr(frame["y"], method=method)


def main():
    pred = pd.read_csv(PRED_H2H)
    actual = pd.read_csv(ACTUAL_H2H)

    pred = pred[["player_id", "H2H_value", "rank_H2H"]].rename(
        columns={
            "H2H_value": "pred_H2H_value",
            "rank_H2H": "pred_rank_H2H",
        }
    )

    actual = actual[["player_id", "H2H_value", "rank_H2H"]].rename(
        columns={
            "H2H_value": "actual_H2H_value",
            "rank_H2H": "actual_rank_H2H",
        }
    )

    pred_ids = set(pred["player_id"])
    actual_ids = set(actual["player_id"])

    pred_only = pred[~pred["player_id"].isin(actual_ids)].copy()
    actual_only = actual[~actual["player_id"].isin(pred_ids)].copy()

    merged = pred.merge(
        actual,
        on="player_id",
        how="inner",
        validate="one_to_one",
    )

    merged["h2h_error"] = merged["pred_H2H_value"] - merged["actual_H2H_value"]
    merged["h2h_abs_error"] = merged["h2h_error"].abs()

    merged["rank_error"] = merged["pred_rank_H2H"] - merged["actual_rank_H2H"]
    merged["rank_abs_error"] = merged["rank_error"].abs()

    # Draft-relevant checks
    merged["actual_top_50"] = merged["actual_rank_H2H"] <= 50
    merged["pred_top_50"] = merged["pred_rank_H2H"] <= 50

    merged["actual_top_100"] = merged["actual_rank_H2H"] <= 100
    merged["pred_top_100"] = merged["pred_rank_H2H"] <= 100

    top50_recall = (
        (merged["actual_top_50"] & merged["pred_top_50"]).sum()
        / max(1, merged["actual_top_50"].sum())
    )

    top50_precision = (
        (merged["actual_top_50"] & merged["pred_top_50"]).sum()
        / max(1, merged["pred_top_50"].sum())
    )

    top100_recall = (
        (merged["actual_top_100"] & merged["pred_top_100"]).sum()
        / max(1, merged["actual_top_100"].sum())
    )

    top100_precision = (
        (merged["actual_top_100"] & merged["pred_top_100"]).sum()
        / max(1, merged["pred_top_100"].sum())
    )

    summary = pd.DataFrame([
        {
            "n_projected": len(pred),
            "n_actual": len(actual),
            "n_overlap_evaluated": len(merged),
            "n_pred_only": len(pred_only),
            "n_actual_only": len(actual_only),

            "h2h_mae": merged["h2h_abs_error"].mean(),
            "h2h_rmse": np.sqrt((merged["h2h_error"] ** 2).mean()),
            "h2h_bias_pred_minus_actual": merged["h2h_error"].mean(),

            "h2h_pearson": corr(
                merged["pred_H2H_value"],
                merged["actual_H2H_value"],
                "pearson",
            ),
            "h2h_spearman": corr(
                merged["pred_H2H_value"],
                merged["actual_H2H_value"],
                "spearman",
            ),

            "rank_mae": merged["rank_abs_error"].mean(),
            "rank_median_abs_error": merged["rank_abs_error"].median(),

            "top50_recall": top50_recall,
            "top50_precision": top50_precision,
            "top100_recall": top100_recall,
            "top100_precision": top100_precision,
        }
    ])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    merged.sort_values("actual_rank_H2H").to_csv(OUT_PLAYER, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    pred_only.to_csv(OUT_DIR / "projected_only_h2h_2026.csv", index=False)
    actual_only.to_csv(OUT_DIR / "actual_only_h2h_2026.csv", index=False)

    print(f"✅ Wrote H2H player-level evaluation -> {OUT_PLAYER}")
    print(f"✅ Wrote H2H summary -> {OUT_SUMMARY}")

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nTop 20 actual H2H players:")
    print(
        merged.sort_values("actual_rank_H2H")[
            [
                "player_id",
                "actual_H2H_value",
                "pred_H2H_value",
                "actual_rank_H2H",
                "pred_rank_H2H",
                "rank_error",
            ]
        ].head(20).to_string(index=False)
    )

    print("\nBiggest rank misses:")
    print(
        merged.sort_values("rank_abs_error", ascending=False)[
            [
                "player_id",
                "actual_H2H_value",
                "pred_H2H_value",
                "actual_rank_H2H",
                "pred_rank_H2H",
                "rank_error",
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()