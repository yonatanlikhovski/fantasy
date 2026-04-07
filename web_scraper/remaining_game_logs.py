import pandas as pd

def check_all():
    df1 = pd.read_csv("done_gamelogs.csv")
    unique_p = df1["player"].unique()
    df2 = pd.read_csv("player_gamelog_urls.csv")
    unique_p2 = df2["player"].unique()
    missing_player = []
    for player in unique_p2:
        if player not in unique_p:
            missing_player.append(player)

    print(missing_player)
    b=0

check_all()
