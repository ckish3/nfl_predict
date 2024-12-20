
from typing import Tuple
import pandas as pd
import numpy as np
from elosports.elo import Elo


def elo_test_score(row, home_field_advantage):
    if row['elo_rating'] + home_field_advantage > row['elo_rating_away'] and row['result'] > 0:
        return 1

    if row['elo_rating'] + home_field_advantage < row['elo_rating_away'] and row['result'] < 0:
        return 1

    return 0

def elo_metric(df_val, elos_df, home_field_advantage):
    df_val = df_val.merge(elos_df, 
                            left_on=['home_team', 'season'],
                            right_on=['team', 'season'])
    df_val = df_val.merge(elos_df, 
                            left_on=['away_team', 'season'],
                            right_on=['team', 'season'],
                            suffixes=('', '_away'))

    df_val['elo_test'] = df_val.apply(elo_test_score, home_field_advantage=home_field_advantage, axis=1)
    
    return df_val[['elo_test']].mean()['elo_test']

def elo_rate(df_games: pd.DataFrame, teams: np.ndarray, k: int=20, home_field: int=100) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produces the Elo rating for each team in each weekin the games data
    Args:
        df_games (pd.DataFrame): A dataframe of games data
        teams (np.ndarray): A numpy array of all unique teams (as strings) that are in the games data
        k (int, optional): The k factor for the Elo rating. Defaults to 20.
        home_field (int, optional): The home field advantage for the Elo rating. Defaults to 100.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of dataframes for training, validation
    """

    team_final_elos = {
                       'season': [],
                       'team': [],
                      'elo_rating': []
        }

    all_elos = {
                'season': [],
                'week': [],
                'team': [],
                'elo_rating': []
        }
    
    
    for season, df_season in df_games.groupby('season'):
        df_season = df_season.sort_values('week', ascending=True)

        eloLeague = Elo(k = k, homefield = home_field)
        for team in teams:
            eloLeague.addPlayer(team)

        for i in df_season.index:
            row = df_season.loc[i]

            is_draw = False
            if row['result'] > 0:
                winner = row['home_team']
                loser = row['away_team']
            elif row['result'] < 0:
                winner = row['away_team']
                loser = row['home_team']
            else:
                is_draw = True
                
            if not is_draw:
                eloLeague.gameOver(winner = winner, 
                                   loser = loser, 
                                   winnerHome=(winner == row['home_team'] and row['location'] == 'Home'))
            
            all_elos['season'].append(season)
            all_elos['week'].append(row['week'])
            all_elos['team'].append(row['home_team'])
            all_elos['elo_rating'].append(eloLeague.ratingDict[row['home_team']])

            all_elos['season'].append(row['season'])
            all_elos['week'].append(row['week'])
            all_elos['team'].append(row['away_team'])
            all_elos['elo_rating'].append(eloLeague.ratingDict[row['away_team']])
            
        for team in teams:
            team_final_elos['team'].append(team)
            team_final_elos['season'].append(season)
            team_final_elos['elo_rating'].append(eloLeague.ratingDict[team])

    df_all_elos = pd.DataFrame(all_elos)
    

    
    elos_df = pd.DataFrame(team_final_elos)

    return df_all_elos, elos_df


def evaluate_elo(df_training, df_val, teams, k, home_field_advantage):
    df_all_elos, elos_df = elo_rate(df_training, teams, k, home_field=home_field_advantage)
    return elo_metric(df_val, elos_df, home_field_advantage)


def simulate_game_result(home_team: str, away_team: str, home_field_advantage: int, elos_df: pd.DataFrame) -> float:
    """
    Returns the probability that the home team will win the game against the away team given the home field advantage that
    the home team has and ELO ratings given by the pandas dataframe elos_df.
    
    Args:
        home_team (str): The home team in the game
        away_team (str): The away team in the game
        home_field_advantage (int): The home field advantage for the Elo rating
        elos_df (pd.DataFrame): A dataframe of Elo ratings for just the latest season

    Returns:
        float: The probability that the home team will win the game
    """

    eloLeague = Elo(k=0) # a dummy k value

    return eloLeague.expectResult(elos_df[elos_df['team'] == home_team]['elo_rating'].values[0] + home_field_advantage,
                                  elos_df[elos_df['team'] == away_team]['elo_rating'].values[0])