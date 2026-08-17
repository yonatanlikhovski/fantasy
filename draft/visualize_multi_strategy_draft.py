# draft/visualize_multi_strategy_draft.py
# Visualize outputs from run_multi_strategy_draft.py
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

SIM_DIR = Path("sim_stats")
OUT_DIR = Path("plots")

STANDINGS_CSV = SIM_DIR / "strategic_draft_standings_2027.csv"
TEAM_AVG_CSV = SIM_DIR / "strategic_draft_team_averages_2027.csv"
WS_PLAYERS_CSV = SIM_DIR / "strategic_draft_win_shares_players_2027.csv"
WS_SUMMARY_CSV = SIM_DIR / "strategic_draft_win_shares_summary_2027.csv"
ROSTERS_CSV = SIM_DIR / "strategic_draft_rosters_2027.csv"

COUNT_CATS = ["PTS", "REB", "AST", "STL", "BLK", "FG3M", "TOV"]
PCT_CATS = ["FG_pct", "FT_pct"]
ALL_CATS = COUNT_CATS + PCT_CATS

STRATEGY_COLORS = {
    "baseline": "#4C78A8",
    "blocks_heavy": "#F58518",
    "guard": "#54A24B",
    "big": "#E45756",
}
DEFAULT_COLOR = "#9D9D9D"


def _team_label(row: pd.Series) -> str:
  punts = row.get("punts", "")
  punt_txt = f" · ignoring {punts}" if isinstance(punts, str) and punts.strip() else ""
  risk = row.get("risk", np.nan)
  risk_txt = f" · risk {risk:.2f}" if pd.notna(risk) else ""
  return f"T{int(row['team'])} {row['strategy']}{punt_txt}{risk_txt}"


def _color_for_strategy(strategy: str) -> str:
  return STRATEGY_COLORS.get(str(strategy), DEFAULT_COLOR)


def _short_label(row: pd.Series) -> str:
  punts = row.get("punts", "")
  if isinstance(punts, str) and punts.strip():
    return f"T{int(row['team'])} ({row['strategy']}, ignoring {punts})"
  return f"T{int(row['team'])} ({row['strategy']})"


def load_data(sim_dir: Path) -> dict[str, pd.DataFrame]:
  paths = {
    "standings": sim_dir / "strategic_draft_standings_2027.csv",
    "team_avg": sim_dir / "strategic_draft_team_averages_2027.csv",
    "ws_players": sim_dir / "strategic_draft_win_shares_players_2027.csv",
    "ws_summary": sim_dir / "strategic_draft_win_shares_summary_2027.csv",
    "rosters": sim_dir / "strategic_draft_rosters_2027.csv",
  }
  missing = [p for p in paths.values() if not p.exists()]
  if missing:
    raise FileNotFoundError(
      "Missing simulation outputs. Run from project root:\n"
      "  python draft/run_multi_strategy_draft.py\n\n"
      + "\n".join(f"  - {p}" for p in missing)
    )
  return {k: pd.read_csv(v) for k, v in paths.items()}


def _apply_style():
  plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#CCCCCC",
    "axes.labelcolor": "#333333",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.8,
    "font.family": "DejaVu Sans",
    "legend.frameon": False,
  })


def plot_standings_leaderboard(standings: pd.DataFrame, out_path: Path):
  df = standings.sort_values("W", ascending=True).copy()
  labels = [_short_label(r) for _, r in df.iterrows()]
  colors = [_color_for_strategy(s) for s in df["strategy"]]

  fig, ax = plt.subplots(figsize=(11, 7))
  y = np.arange(len(df))
  bars = ax.barh(y, df["W"], color=colors, height=0.62, edgecolor="white", linewidth=0.8)

  for i, (bar, cat_pts) in enumerate(zip(bars, df["cat_pts"])):
    ax.text(
      bar.get_width() + 0.08,
      bar.get_y() + bar.get_height() / 2,
      f"{cat_pts:.1f} cat pts",
      va="center",
      ha="left",
      fontsize=9,
      color="#555555",
    )

  ax.set_yticks(y)
  ax.set_yticklabels(labels, fontsize=9)
  ax.set_xlabel("Expected weekly wins (250-trial sim)")
  ax.set_title("Projected Standings by Strategy")
  ax.set_xlim(0, df["W"].max() * 1.22)
  ax.grid(axis="x", alpha=0.5)

  handles = [
    mpatches.Patch(color=c, label=s.replace("_", " ").title())
    for s, c in STRATEGY_COLORS.items()
  ]
  ax.legend(handles=handles, loc="lower right", title="Strategy")

  fig.tight_layout()
  fig.savefig(out_path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def plot_performance_scatter(standings: pd.DataFrame, out_path: Path):
  fig, ax = plt.subplots(figsize=(9, 7))
  for strategy, grp in standings.groupby("strategy"):
    ax.scatter(
      grp["W"],
      grp["cat_pts"],
      s=70 + grp["risk"].fillna(0.1) * 420,
      c=_color_for_strategy(strategy),
      alpha=0.88,
      edgecolors="white",
      linewidths=0.9,
      label=strategy.replace("_", " ").title(),
      zorder=3,
    )
    for _, row in grp.iterrows():
      ax.annotate(
        f"T{int(row['team'])}",
        (row["W"], row["cat_pts"]),
        textcoords="offset points",
        xytext=(5, 4),
        fontsize=8,
        color="#333333",
      )

  ax.set_xlabel("Expected weekly wins")
  ax.set_ylabel("Category points (wins + 0.5×ties)")
  ax.set_title("Win Rate vs Category Dominance\n(marker size = risk aversion)")
  ax.grid(alpha=0.45)
  ax.legend(title="Strategy", loc="lower right")
  fig.tight_layout()
  fig.savefig(out_path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def plot_category_heatmap(team_avg: pd.DataFrame, standings: pd.DataFrame, out_path: Path):
  merged = team_avg.merge(standings[["team", "strategy", "punts", "risk"]], on="team", how="left")
  merged = merged.sort_values("team")

  mat = merged[ALL_CATS].copy()
  for c in COUNT_CATS:
    if c == "TOV":
      mat[c] = -mat[c]
  z = (mat - mat.mean()) / mat.std(ddof=0).replace(0, 1)

  labels = [_short_label(r) for _, r in merged.iterrows()]

  fig, ax = plt.subplots(figsize=(12, 7.5))
  im = ax.imshow(z.values, aspect="auto", cmap="RdYlGn", vmin=-1.8, vmax=1.8)

  ax.set_xticks(range(len(ALL_CATS)))
  ax.set_xticklabels(
    [c.replace("_pct", "%").replace("FG3M", "3PM") for c in ALL_CATS],
    rotation=35,
    ha="right",
  )
  ax.set_yticks(range(len(labels)))
  ax.set_yticklabels(labels, fontsize=8.5)
  ax.set_title("Team Category Profile (z-score vs league average)\nTOV inverted so green = better")

  cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
  cbar.set_label("Std dev from mean")

  fig.tight_layout()
  fig.savefig(out_path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def _radar_angles(n: int) -> np.ndarray:
  angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
  return np.concatenate([angles, angles[:1]])


def plot_top_teams_radar(team_avg: pd.DataFrame, standings: pd.DataFrame, out_path: Path, top_n: int = 4):
  merged = team_avg.merge(standings, on="team", how="left")
  merged = merged.sort_values("W", ascending=False).head(top_n)

  radar_cats = ["PTS", "REB", "AST", "STL", "BLK", "FG3M", "FG_pct", "FT_pct"]
  pool = team_avg[radar_cats].copy()
  pool["TOV_inv"] = -team_avg["TOV"]
  radar_cats_plot = radar_cats + ["TOV_inv"]
  pool_vals = pool[radar_cats_plot]
  mins = pool_vals.min()
  maxs = pool_vals.max()
  span = (maxs - mins).replace(0, 1)

  angles = _radar_angles(len(radar_cats_plot))
  tick_labels = [c.replace("_pct", "%").replace("FG3M", "3PM").replace("TOV_inv", "TOV⁻") for c in radar_cats_plot]

  fig = plt.figure(figsize=(11, 8))
  gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

  for ax_idx, (_, row) in enumerate(merged.iterrows()):
    ax = fig.add_subplot(gs[ax_idx // 2, ax_idx % 2], polar=True)
    vals = row[radar_cats].tolist() + [-(row["TOV"])]
    normed = ((pd.Series(vals, index=radar_cats_plot) - mins) / span).tolist()
    vals_closed = normed + normed[:1]

    color = _color_for_strategy(row["strategy"])
    ax.plot(angles, vals_closed, color=color, linewidth=2.2)
    ax.fill(angles, vals_closed, color=color, alpha=0.18)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.05)
    ax.set_title(_short_label(row), fontsize=10, pad=14)
    ax.grid(alpha=0.35)

  fig.suptitle("Top Teams — Normalized Category Shape", fontsize=14, fontweight="bold", y=1.02)
  fig.savefig(out_path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def plot_win_shares_summary(ws_summary: pd.DataFrame, out_path: Path):
  df = ws_summary.sort_values("team_win_shares_sum", ascending=True).copy()
  labels = [_short_label(r) for _, r in df.iterrows()]
  colors = [_color_for_strategy(s) for s in df["strategy"]]

  fig, ax = plt.subplots(figsize=(10, 7))
  ax.barh(labels, df["team_win_shares_sum"], color=colors, edgecolor="white", linewidth=0.8)
  ax.set_xlabel("Total player win shares (season sum)")
  ax.set_title("Aggregate Win Shares by Team")
  ax.grid(axis="x", alpha=0.45)
  fig.tight_layout()
  fig.savefig(out_path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def plot_player_ws_distribution(ws_players: pd.DataFrame, standings: pd.DataFrame, out_path: Path):
  merged = ws_players.drop(columns=["strategy", "punts", "risk"], errors="ignore").merge(
    standings[["team", "strategy", "punts", "risk", "W"]], on="team", how="left"
  )
  order = (
    merged.groupby("team")["win_shares"]
    .sum()
    .sort_values(ascending=False)
    .index.tolist()
  )
  labels_map = {int(r["team"]): _short_label(r) for _, r in standings.iterrows()}

  data = [merged.loc[merged["team"] == t, "win_shares"].values for t in order]
  colors = [_color_for_strategy(merged.loc[merged["team"] == t, "strategy"].iloc[0]) for t in order]

  fig, ax = plt.subplots(figsize=(12, 6.5))
  bp = ax.boxplot(
    data,
    vert=True,
    patch_artist=True,
    widths=0.55,
    medianprops={"color": "#222222", "linewidth": 1.6},
    whiskerprops={"color": "#888888"},
    capprops={"color": "#888888"},
    flierprops={"marker": "o", "markersize": 4, "alpha": 0.45},
  )
  for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.35)
    patch.set_edgecolor(color)

  for i, team_vals in enumerate(data, start=1):
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(team_vals))
    ax.scatter(
      np.full(len(team_vals), i) + jitter,
      team_vals,
      s=22,
      c=colors[i - 1],
      alpha=0.75,
      edgecolors="white",
      linewidths=0.4,
      zorder=3,
    )

  ax.set_xticks(range(1, len(order) + 1))
  ax.set_xticklabels([labels_map[int(t)] for t in order], rotation=35, ha="right", fontsize=8)
  ax.set_ylabel("Player win shares")
  ax.set_title("Roster Win-Share Distribution (box + individual players)")
  ax.grid(axis="y", alpha=0.4)
  fig.tight_layout()
  fig.savefig(out_path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def plot_top_players(ws_players: pd.DataFrame, standings: pd.DataFrame, out_path: Path, top_n: int = 20):
  top = ws_players.nlargest(top_n, "win_shares").copy()
  top = top.drop(columns=["strategy", "punts"], errors="ignore").merge(
    standings[["team", "strategy", "punts"]], on="team", how="left"
  )
  top["label"] = top.apply(
    lambda r: f"{r['player_id']} (T{int(r['team'])} {r['strategy']})", axis=1
  )
  top = top.sort_values("win_shares", ascending=True)
  colors = [_color_for_strategy(s) for s in top["strategy"]]

  fig, ax = plt.subplots(figsize=(10, 8))
  ax.barh(top["label"], top["win_shares"], color=colors, edgecolor="white", linewidth=0.7)
  ax.set_xlabel("Win shares")
  ax.set_title(f"Top {top_n} Players by Simulated Win Shares")
  ax.grid(axis="x", alpha=0.4)
  fig.tight_layout()
  fig.savefig(out_path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def plot_draft_board(rosters: pd.DataFrame, ws_players: pd.DataFrame, out_path: Path, n_teams: int = 10, rounds: int = 13, seed: int = 123):
  rng = np.random.default_rng(seed)
  order = list(range(n_teams))
  rng.shuffle(order)
  pick_sequence = []
  for r in range(rounds):
    pick_sequence.extend(order if r % 2 == 0 else list(reversed(order)))

  ws_map = ws_players.set_index(["team", "player_id"])["win_shares"].to_dict()
  team_players = {t: [] for t in range(1, n_teams + 1)}
  for _, row in rosters.iterrows():
    team_players[int(row["team"])].append(row["player_id"])

  picks = []
  for pick_no, team_idx in enumerate(pick_sequence, start=1):
    roster = team_players[team_idx + 1]
    if not roster:
      continue
    pid = roster.pop(0)
    ws = ws_map.get((team_idx + 1, pid), 0.0)
    picks.append({
      "pick": pick_no,
      "round": (pick_no - 1) // n_teams + 1,
      "team": team_idx + 1,
      "player_id": pid,
      "win_shares": ws,
    })

  picks_df = pd.DataFrame(picks)
  pivot = picks_df.pivot(index="round", columns="team", values="win_shares")

  fig, ax = plt.subplots(figsize=(12, 7))
  im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0)

  ax.set_xticks(range(n_teams))
  ax.set_xticklabels([f"T{i}" for i in range(1, n_teams + 1)])
  ax.set_yticks(range(rounds))
  ax.set_yticklabels([f"R{r}" for r in range(1, rounds + 1)])
  ax.set_xlabel("Team")
  ax.set_ylabel("Draft round")
  ax.set_title("Draft Board Heatmap — Player Win Shares by Pick")

  for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
      val = pivot.values[i, j]
      if np.isfinite(val):
        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color="#222222")

  cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
  cbar.set_label("Win shares")
  fig.tight_layout()
  fig.savefig(out_path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def plot_strategy_comparison(standings: pd.DataFrame, ws_summary: pd.DataFrame, out_path: Path):
  merged = standings.merge(
    ws_summary[["team", "team_win_shares_sum"]],
    on="team",
    how="left",
  )
  merged["build"] = merged.apply(
    lambda r: f"{r['strategy']}" + (f"\n(ignoring {r['punts']})" if isinstance(r['punts'], str) and r['punts'].strip() else ""),
    axis=1,
  )

  fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
  metrics = [
    ("W", "Expected wins", "#4C78A8"),
    ("cat_pts", "Category points", "#54A24B"),
    ("team_win_shares_sum", "Total win shares", "#F58518"),
  ]
  x = np.arange(len(merged))
  for ax, (col, title, color) in zip(axes, metrics):
    vals = merged.sort_values("team")[col]
    teams = merged.sort_values("team")["team"]
    bar_colors = [_color_for_strategy(s) for s in merged.sort_values("team")["strategy"]]
    ax.bar(teams.astype(str), vals, color=bar_colors, edgecolor="white", linewidth=0.7)
    ax.set_xlabel("Team")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.4)

  fig.suptitle("Per-Team Performance Metrics", fontsize=14, fontweight="bold")
  fig.tight_layout()
  fig.savefig(out_path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def write_html_dashboard(out_dir: Path, plot_files: list[tuple[str, str]]):
  cards = "\n".join(
    f"""
    <section class="card">
      <h2>{title}</h2>
      <img src="{fname}" alt="{title}" loading="lazy" />
    </section>"""
    for title, fname in plot_files
  )

  html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Multi-Strategy Draft Dashboard</title>
  <style>
    :root {{
      --bg: #f4f5f7;
      --card: #ffffff;
      --text: #1f2933;
      --muted: #6b7280;
      --accent: #4c78a8;
      --border: #e5e7eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    header {{
      padding: 2rem 2rem 1rem;
      border-bottom: 1px solid var(--border);
      background: var(--card);
    }}
    header h1 {{ margin: 0 0 0.35rem; font-size: 1.65rem; }}
    header p {{ margin: 0; color: var(--muted); max-width: 62rem; }}
    main {{
      padding: 1.5rem;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 1.25rem;
      max-width: 1500px;
      margin: 0 auto;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem 1rem 1.1rem;
    }}
    .card h2 {{
      margin: 0 0 0.75rem;
      font-size: 1rem;
      color: var(--accent);
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 6px;
      border: 1px solid var(--border);
    }}
    @media (max-width: 520px) {{
      main {{ grid-template-columns: 1fr; padding: 1rem; }}
      header {{ padding: 1.25rem; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Multi-Strategy Fantasy Draft — 2027 Simulation</h1>
    <p>
      Visual summary of <code>run_multi_strategy_draft.py</code> outputs:
      10 teams with distinct ignoring/build strategies, snake draft, and fast win-share Monte Carlo (250 trials).
    </p>
  </header>
  <main>
    {cards}
  </main>
</body>
</html>
"""
  (out_dir / "index.html").write_text(html, encoding="utf-8")


def main():
  ap = argparse.ArgumentParser(description="Visualize multi-strategy draft simulation outputs")
  ap.add_argument("--sim-dir", type=Path, default=SIM_DIR)
  ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
  ap.add_argument("--seed", type=int, default=123, help="Draft seed (must match simulation)")
  ap.add_argument("--teams", type=int, default=10)
  ap.add_argument("--rounds", type=int, default=13)
  args = ap.parse_args()

  _apply_style()
  args.out_dir.mkdir(parents=True, exist_ok=True)

  data = load_data(args.sim_dir)
  standings = data["standings"]
  team_avg = data["team_avg"]
  ws_players = data["ws_players"]
  ws_summary = data["ws_summary"]
  rosters = data["rosters"]

  plots: list[tuple[str, str, Path]] = [
    ("Projected Standings", "01_standings_leaderboard.png", args.out_dir / "01_standings_leaderboard.png"),
    ("Win Rate vs Category Points", "02_performance_scatter.png", args.out_dir / "02_performance_scatter.png"),
    ("Category Profile Heatmap", "03_category_heatmap.png", args.out_dir / "03_category_heatmap.png"),
    ("Top Teams Radar", "04_top_teams_radar.png", args.out_dir / "04_top_teams_radar.png"),
    ("Win Shares by Team", "05_win_shares_summary.png", args.out_dir / "05_win_shares_summary.png"),
    ("Roster WS Distribution", "06_player_ws_distribution.png", args.out_dir / "06_player_ws_distribution.png"),
    ("Top Players", "07_top_players.png", args.out_dir / "07_top_players.png"),
    ("Draft Board", "08_draft_board.png", args.out_dir / "08_draft_board.png"),
    ("Per-Team Metrics", "09_strategy_comparison.png", args.out_dir / "09_strategy_comparison.png"),
  ]

  plot_standings_leaderboard(standings, plots[0][2])
  plot_performance_scatter(standings, plots[1][2])
  plot_category_heatmap(team_avg, standings, plots[2][2])
  plot_top_teams_radar(team_avg, standings, plots[3][2])
  plot_win_shares_summary(ws_summary, plots[4][2])
  plot_player_ws_distribution(ws_players, standings, plots[5][2])
  plot_top_players(ws_players, standings, plots[6][2])
  plot_draft_board(rosters, ws_players, plots[7][2], n_teams=args.teams, rounds=args.rounds, seed=args.seed)
  plot_strategy_comparison(standings, ws_summary, plots[8][2])

  write_html_dashboard(args.out_dir, [(t, f) for t, f, _ in plots])

  print("\nMulti-strategy draft visualizations saved:")
  for title, fname, path in plots:
    print(f"  {path}")
  print(f"\nDashboard: {args.out_dir / 'index.html'}")
  print("\nOpen the HTML file in a browser for the full report.")


if __name__ == "__main__":
  main()
