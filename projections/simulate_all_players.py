# simulate_all_players.py
import pandas as pd
from monte_carlo_sim import simulate_many_seasons


FEATURES_CSV = "sim_stats/player_features_train_2023_2026.csv"
OUT_CSV = "sim_stats/projected_2027_weekly.csv"

# Use real player names here, because player_id is now the real name.
EXCLUDED_PLAYERS = {
    "Donte DiVincenzo"
}


def normalize_player_name(name: str) -> str:
    """
    Normalize names so exclusion matching is not broken by spaces,
    capitalization, or Basketball-Reference trailing stars.
    """
    return str(name).strip().rstrip("*").lower()


def main():
    # Load features CSV
    df = pd.read_csv(FEATURES_CSV)

    if "player_id" not in df.columns:
        raise KeyError("Feature CSV must contain a 'player_id' column.")

    excluded_normalized = {
        normalize_player_name(name)
        for name in EXCLUDED_PLAYERS
    }

    if excluded_normalized:
        before = len(df)

        df["_exclude_key"] = df["player_id"].map(normalize_player_name)

        excluded_df = df[df["_exclude_key"].isin(excluded_normalized)].copy()

        if not excluded_df.empty:
            print("\nExcluded players:")
            print(excluded_df["player_id"].to_string(index=False))

        df = df[~df["_exclude_key"].isin(excluded_normalized)].copy()
        df = df.drop(columns=["_exclude_key"])

        after = len(df)
        print(f"\nExcluded {before - after} players from simulation.")
    else:
        print("\nNo excluded players.")

    results = []

    for idx, (_, row) in enumerate(df.iterrows()):
        player = row.to_dict()
 
        # Run simulation
        agg, ci, _ = simulate_many_seasons(
            player,
            n_seasons=500,
            games_in_season=82,
            games_per_week=3.5,
            seed=42 + idx * 1009,
        )

        # Flatten into a single row
        out = {"player_id": player["player_id"]}

        # Keep source_file_id if it exists, useful for debugging.
        if "source_file_id" in player:
            out["source_file_id"] = player["source_file_id"]

        for k, v in agg.items():
            if k != "player_id":
                out[f"{k}_mean"] = v

        for k, (lo, hi) in ci.items():
            out[f"{k}_ci_lo"] = lo
            out[f"{k}_ci_hi"] = hi

        results.append(out)

    # Save to CSV
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"\n✅ Simulated {len(results)} players → {OUT_CSV}")


if __name__ == "__main__":
    main()