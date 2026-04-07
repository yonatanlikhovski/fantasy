# run_all_players.py
import pandas as pd
from monte_carlo_sim import simulate_many_seasons

def main():
    # Load features CSV (already built by build_player_features_summary.py)
    df = pd.read_csv("sim_stats/player_features_summary.csv")

    results = []

    for _, row in df.iterrows():
        player = row.to_dict()

        # Run simulation
        agg, ci, _ = simulate_many_seasons(
            player,
            n_seasons=1000,
            games_in_season=82,
            games_per_week=3.5,
            seed=42
        )
       
        # Flatten into a single row
        out = {"player_id": player["player_id"]}
        for k, v in agg.items():
            if k != "player_id":
                out[f"{k}_mean"] = v
        for k, (lo, hi) in ci.items():
            out[f"{k}_ci_lo"] = lo
            out[f"{k}_ci_hi"] = hi

        results.append(out)

    # Save to CSV
    out_df = pd.DataFrame(results)
    out_df.to_csv("sim_stats/weekly_sim_results.csv", index=False)
    print(f"✅ Simulated {len(results)} players → sim_stats/weekly_sim_results.csv")

if __name__ == "__main__":
    main()
