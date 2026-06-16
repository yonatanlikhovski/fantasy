import math
import numpy as np
import pandas as pd
import random as rand
import numpy as np



def plays_game(durability, games_in_season=82):
    num = rand.randint(0,games_in_season-1)
    return num < durability

def stats_sim(mean, std):
    val = rand.gauss(mean, std)   # sample from Normal(mean, std)
    val = max(0, val)               # stats can't go below 0
    return round(val)               # round to nearest int (optional)

def get_float(player, key, default=0.0):
    try:
        val = player.get(key, default)
    except AttributeError:
        val = default

    if val is None or pd.isna(val):
        return default

    return float(val)


def stats_sim_from_role(player, stat, role):
    """
    Simulate a stat using role × per-role rate when available.
    Falls back to the old raw mean/std method.
    """
    rate_key = f"{stat}_per_role"
    mean_key = f"{stat}_mean"
    std_key = f"{stat}_std"

    rate = get_float(player, rate_key, default=np.nan)

    if not np.isnan(rate):
        mean = rate * role

        base_std = get_float(player, std_key, default=0.5)
        base_role = max(1.0, get_float(player, "ROLE_mean", default=role))

        # Scale uncertainty with role
        std = base_std * math.sqrt(max(1.0, role) / base_role)

        return stats_sim(mean, std)

    return stats_sim(
        get_float(player, mean_key, default=0.0),
        get_float(player, std_key, default=0.5),
    )

def simulate_game(player):
    """
    Simulate one game for the given player dict.
    Uses role/opportunity proxy when available.
    """

    if not plays_game(player["durability"]):
        return {
            "player_id": player["player_id"],
            "played": False,
            "ROLE": 0,
            "PTS": 0,
            "REB": 0,
            "AST": 0,
            "STL": 0,
            "BLK": 0,
            "TOV": 0,
            "FG3M": 0,
            "FGA": 0,
            "FGM": 0,
            "FG%": None,
            "FTA": 0,
            "FTM": 0,
            "FT%": None,
        }

    # Hidden role / opportunity for this game
    role = stats_sim(
        get_float(player, "ROLE_mean", default=0.0),
        get_float(player, "ROLE_std", default=0.0),
    )

    # Counting stats from role × rate
    pts = stats_sim_from_role(player, "PTS", role)
    reb = stats_sim_from_role(player, "REB", role)
    ast = stats_sim_from_role(player, "AST", role)
    stl = stats_sim_from_role(player, "STL", role)
    blk = stats_sim_from_role(player, "BLK", role)
    tov = stats_sim_from_role(player, "TOV", role)
    fg3m = stats_sim_from_role(player, "FG3M", role)

    # Shooting attempts from role × attempt rate
    fga = stats_sim_from_role(player, "FGA", role)
    fgp = max(0, min(1, rand.gauss(
        get_float(player, "FGP_mean", default=0.45),
        get_float(player, "FGP_std", default=0.05),
    )))
    fgm = np.random.binomial(fga, fgp) if fga > 0 else 0

    fta = stats_sim_from_role(player, "FTA", role)
    ftp = max(0, min(1, rand.gauss(
        get_float(player, "FTP_mean", default=0.78),
        get_float(player, "FTP_std", default=0.05),
    )))
    ftm = np.random.binomial(fta, ftp) if fta > 0 else 0

    return {
        "player_id": player["player_id"],
        "played": True,
        "ROLE": role,
        "PTS": pts,
        "REB": reb,
        "AST": ast,
        "STL": stl,
        "BLK": blk,
        "TOV": tov,
        "FG3M": fg3m,
        "FGA": fga,
        "FGM": fgm,
        "FG%": fgm / fga if fga else None,
        "FTA": fta,
        "FTM": ftm,
        "FT%": ftm / fta if fta else None,
    }


# use the same stat keys your simulate_game returns
COUNTING_KEYS = ["ROLE","PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FGM", "FGA", "FTM", "FTA"]

def simulate_season(player, games_in_season=82, games_per_week=3.5):
    """
    Uses your simulate_game(player) N times, sums totals, counts games played,
    and converts season totals to a weekly estimate by dividing by (82/3.5).
    """
    season_totals = {k: 0 for k in COUNTING_KEYS}
    games_played = 0
    logs = []

    for _ in range(games_in_season):
        g = simulate_game(player)          # <-- your existing function
        games_played += int(bool(g.get("played", False)))
        for k in COUNTING_KEYS:
            season_totals[k] += int(g.get(k, 0))
        logs.append(g)

    # derive season-level percentages from totals (how H2H % cats work)
    fg_pct = (season_totals["FGM"] / season_totals["FGA"]) if season_totals["FGA"] > 0 else None
    ft_pct = (season_totals["FTM"] / season_totals["FTA"]) if season_totals["FTA"] > 0 else None

    # convert season totals -> weekly estimate
    weekly_scale = games_in_season / games_per_week  # 82 / 3.5
    weekly_est = {k: season_totals[k] / weekly_scale for k in COUNTING_KEYS}
    weekly_est["FG%"] = fg_pct
    weekly_est["FT%"] = ft_pct

    return {
        "player_id": player["player_id"],
        "games_played": games_played,
        "season_totals": season_totals,
        "season_fg_pct": fg_pct,
        "season_ft_pct": ft_pct,
        "weekly_estimate": weekly_est,
        "game_logs": logs,  # keep if you want to inspect single games
    }

def simulate_many_seasons(player, n_seasons=100, games_in_season=82, games_per_week=3.5, seed=None):
    """
    Repeats simulate_season() n times and aggregates weekly estimates.
    Keeps your structure; no changes to simulate_game/stats_sim/plays_game.
    """
    if seed is not None:
        np.random.seed(seed)
        rand.seed(seed)
    weekly_rows = []
    for _ in range(n_seasons):
        res = simulate_season(player, games_in_season, games_per_week)
        row = res["weekly_estimate"].copy()
        row["games_played"] = res["games_played"]
        weekly_rows.append(row)

    # mean + 5/95% CI per weekly stat
    agg = {}
    ci = {}
    for k in weekly_rows[0].keys():
        vals = [r[k] for r in weekly_rows if r[k] is not None]
        if not vals:
            agg[k] = None
            ci[k] = (None, None)
        else:
            arr = np.array(vals, dtype=float)
            agg[k] = float(arr.mean())
            ci[k]  = (float(np.percentile(arr, 5)), float(np.percentile(arr, 95)))

    agg["player_id"] = player["player_id"]
    return agg, ci, weekly_rows
