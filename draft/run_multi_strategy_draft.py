# run_multi_strategy_draft.py
# Use different strategies per team, draft, and simulate fast Win Shares.
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# Import functions from your existing draft/compute_win_shares.py
from compute_win_shares import (
    load_player_pool,
    compute_draft_ranking,
    compute_ws_fast,           # the fast version you added
    team_averages,
    strategy_weights,
    apply_punts,
)

OUT_DIR = Path("sim_stats")

def make_snake_order(n_teams: int, rounds: int, seed: int):
    rng = np.random.default_rng(seed)
    order = list(range(n_teams))
    rng.shuffle(order)
    seq = []
    for r in range(rounds):
        seq.extend(order if r % 2 == 0 else reversed(order))
    return seq

def pick_best_available(df: pd.DataFrame, available: set[str], weights: dict, risk: float) -> str:
    # rank only among available players using this team’s weights & risk
    avail_df = df[df["player_id"].isin(available)].copy()
    if avail_df.empty:
        return None
    ranked = compute_draft_ranking(avail_df, risk_aversion=risk, weights=weights)
    return ranked["player_id"].iloc[0]

def main():
    ap = argparse.ArgumentParser(description="10-team multi-strategy draft + fast WS sim")
    ap.add_argument("--teams", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=13)
    ap.add_argument("--trials", type=int, default=250)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--gpw", type=float, default=3.5)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------- Load player pool -------------
    df = load_player_pool(
        Path("sim_stats/projected_2027_weekly.csv"),
        Path("sim_stats/player_features_train_2023_2026.csv"),
        args.gpw
    )   
    # ------------- Define team strategies -------------
    # 9 themed strategies + team 10 is SOLID (balanced, safer picks)
    # You can tweak any weight or punt set below.
    strat_plan = [
        {"name": "baseline",      "punts": set(),            "risk": 0.10},  # Team 1
        {"name": "blocks_heavy",  "punts": set(),            "risk": 0.10},  # Team 2
        {"name": "guard",         "punts": {"REB","BLK"},    "risk": 0.10},  # Team 3
        {"name": "big",           "punts": {"AST","FG3M"},   "risk": 0.10},  # Team 4
        {"name": "baseline",      "punts": {"FT"},           "risk": 0.10},  # Team 5  (punt FT)
        {"name": "baseline",      "punts": {"TOV"},          "risk": 0.10},  # Team 6  (punt TOV)
        {"name": "guard",         "punts": {"FT"},           "risk": 0.10},  # Team 7  (threes/assists/steals; punt FT)
        {"name": "big",           "punts": {"FT"},           "risk": 0.10},  # Team 8  (boards/blocks/FG; punt FT)
        {"name": "baseline",      "punts": {"PTS"},          "risk": 0.10},  # Team 9  (de-emphasize raw scoring)
        {"name": "baseline",      "punts": set(),            "risk": 0.28},  # Team 10 (SOLID: balanced + low-risk)
    ]

    assert args.teams == len(strat_plan), "Adjust strat_plan length to match --teams"

    # Precompute weight dicts per team
    team_weights = []
    for s in strat_plan:
        w = strategy_weights(s["name"])
        w = apply_punts(w, s["punts"])
        team_weights.append({"weights": w, "risk": float(s["risk"]), "label": s["name"], "punts": ",".join(sorted(s["punts"]))})

    # ------------- Snake draft (per-team strategy) -------------
    rosters = {t: [] for t in range(args.teams)}
    available = set(df["player_id"].tolist())

    pick_sequence = make_snake_order(args.teams, args.rounds, args.seed)
    for pick_idx, team in enumerate(pick_sequence, start=1):
        strat = team_weights[team]
        pid = pick_best_available(df, available, weights=strat["weights"], risk=strat["risk"])
        if pid is None: break
        rosters[team].append(pid)
        available.remove(pid)

    # Save rosters with strategy label for transparency
    r_rows = []
    for t in range(args.teams):
        meta = team_weights[t]
        for p in rosters[t]:
            r_rows.append({"team": t+1, "player_id": p, "strategy": meta["label"], "punts": meta["punts"], "risk": meta["risk"]})
    rosters_df = pd.DataFrame(r_rows)
    rosters_df.to_csv(OUT_DIR / "strategic_draft_rosters_2027.csv", index=False)

    # ------------- Inspect deterministic team averages -------------
    team_avg = team_averages(df, rosters)
    team_avg["team"] = team_avg["team"] + 1
    team_avg.to_csv(OUT_DIR / "strategic_draft_team_averages_2027.csv", index=False)

    # ------------- Fast WS simulation -------------
    standings, ws_players = compute_ws_fast(df, rosters, trials=args.trials, seed=args.seed)
    standings["team"] = standings["team"] + 1
    ws_players["team"] = ws_players["team"] + 1

    # Attach strategy labels to outputs
    strat_df = pd.DataFrame([{"team": i+1, "strategy": team_weights[i]["label"], "punts": team_weights[i]["punts"], "risk": team_weights[i]["risk"]} for i in range(args.teams)])
    standings = standings.merge(strat_df, on="team", how="left")
    ws_players = ws_players.merge(strat_df, on="team", how="left")

    standings = standings.sort_values(["W","cat_pts"], ascending=[False, False]).reset_index(drop=True)

    # ------------- Save -------------
    standings.to_csv(OUT_DIR / "strategic_draft_standings_2027.csv", index=False)
    ws_players.to_csv(OUT_DIR / "strategic_draft_win_shares_players_2027.csv", index=False)

    # Team WS totals
    ws_summary = ws_players.groupby(["team","strategy","punts","risk"], as_index=False)["win_shares"].sum().rename(columns={"win_shares":"team_win_shares_sum"})
    ws_summary = ws_summary.sort_values("team_win_shares_sum", ascending=False)
    ws_summary.to_csv(OUT_DIR / "strategic_draft_win_shares_summary_2027.csv", index=False)

    # ------------- Console peek -------------
    print("\n=== Multi-strategy draft: team assignments ===")
    for i in range(args.teams):
        s = team_weights[i]
        print(f"Team {i+1}: {s['label']}  | punts: {s['punts'] or '—'}  | risk: {s['risk']:.2f}")

    print("\n=== Expected standings (fast WS sim) ===")
    print(standings.to_string(index=False))

    print("\nSaved:")
    print("  sim_stats/strategic_draft_rosters.csv")
    print("  sim_stats/strategic_draft_team_averages.csv")
    print("  sim_stats/strategic_draft_standings.csv")
    print("  sim_stats/strategic_draft_win_shares_players.csv")
    print("  sim_stats/strategic_draft_win_shares_summary.csv")

if __name__ == "__main__":
    main()
