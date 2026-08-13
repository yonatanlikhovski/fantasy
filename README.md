**NBA Fantasy Value Prediction**

A data-driven model for projecting NBA player value in 9-category Head-to-Head fantasy basketball.

Using NBA game data from the 2023–2026 seasons, the project estimates player performance for 2027, models uncertainty through Monte Carlo simulation, converts projected statistics into fantasy value, and tests different draft strategies.

**Pipeline**

1. Build player profiles from historical game logs using recent performance, variability, role, and availability.
2. Project the 2027 season by simulating 500 possible seasons for each player.
3. Calculate H2H fantasy value across PTS, REB, AST, STL, BLK, 3PM, TOV, FG%, and FT%.
4. Simulate fantasy drafts using balanced and category-focused strategies.
5. Evaluate the model through a historical backtest comparing 2026 projections with actual 2026 performance.

**Main Outputs**

- sim_stats/h2h_value_2027.csv — projected 2027 player rankings
- sim_stats/projected_2027_weekly.csv — projected weekly statistics
- sim_stats/strategic_draft_rosters_2027.csv — simulated draft rosters
- sim_stats/strategic_draft_standings_2027.csv — simulated league results
- sim_stats/draft_win_shares_players_2027.csv — estimated player contribution to team success
- sim_stats/evaluation_2026_summary.csv — historical projection evaluation

**Repository Structure**

- web_scraper/ — Basketball Reference data collection
- projections/ — player profiles, projections, and Monte Carlo simulation
- draft/ — H2H valuation and draft simulations
- sim_stats/ — processed data and model outputs
- visual_helper/ — exploratory visualizations
- Evaluation

The projection model is backtested by using data available before the 2026 season to predict 2026 performance, then comparing those projections with the actual season results. The evaluation includes prediction error, ranking correlation, and confidence-interval coverag