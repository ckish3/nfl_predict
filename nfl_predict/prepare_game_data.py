
import re
import pandas as pd
import numpy as np
import random

from elosports.elo import Elo


def download_data() -> pd.DataFrame:
    """
    Download past (from 1999 onward)& current season upcoming game data from http://www.habitatring.com/games.csv
    
    Returns:
        pd.DataFrame
    """
    return pd.read_csv(r'http://www.habitatring.com/games.csv')


def get_historical_games(full_data: pd.DataFrame, game_type='REG') -> pd.DataFrame:
    """

    Args:
        full_data (pd.DataFrame): A dataframe of all game data; both historical and upcoming

    Returns:
        pd.DataFrame: Just the historical (completed) games
    """
    df = full_data[pd.notnull(full_data['away_score'])]
    
    if game_type is not None:
        df = df[df['game_type'] == game_type]

    return df


def get_all_teams(games_df: pd.DataFrame) -> np.ndarray:
    """

    Args:
        games_df (pd.DataFrame): A dataframe of games data

    Returns:
        np.ndarray: A numpy array of all unique teams (as strings) that are in the games data
    """
    return games_df['home_team'].unique()


def create_team_histories(games_df: pd.DataFrame) -> pd.DataFrame:
    """

    Args:
        games_df (pd.DataFrame): A dataframe of games data

    Returns:
        pd.DataFrame: A reformatting of the games data where each team has all of its games.
            So this dataframe will be twice as long as the original because each game will 
            appear twice, once for the home team and once for the away team
    """

    teams_dict = {}

    for i in games_df.index:
        row = games_df.loc[i]
        teams_dict = add_game_to_team_dict(teams_dict, row, False)
        teams_dict = add_game_to_team_dict(teams_dict, row, True)

    return add_record(games_df, teams_dict)
    
    
def add_game_to_team_dict(team_dict, row, is_home_team):
    if is_home_team:
        prefix = 'home'
        opponent_prefix = 'away'
        spread_factor = 1
    else:
        prefix = 'away'
        opponent_prefix = 'home'
        spread_factor = -1

    team = row[prefix + '_team']
        
    if team not in team_dict:
        team_dict[team] = {
                            'team': [],
                            'season': [],
                            'week': [],
                            'opponent': [],
                            'wins': [],
                            'losses': [],
                            'draws': [],
                            'wins_before': [],
                            'losses_before': [],
                            'draws_before': [],
                            
                            }
    team_dict[team]['team'].append(team)
    team_dict[team]['season'].append(row['season'])
    team_dict[team]['week'].append(row['week'])
    team_dict[team]['opponent'].append(row[opponent_prefix + '_team'])

    points = row[prefix + '_score']
    opponent_points = row[opponent_prefix + '_score']

    win = 0
    loss = 0
    draw = 0
    
    if points > opponent_points:
        result = 'win'
        win = 1
    elif points < opponent_points:
        result = 'lose'
        loss = 1
    else:
        result = 'draw'
        draw = 1

    if len(team_dict[team]['wins']) == 0:
        team_dict[team]['wins'].append(win)
        team_dict[team]['losses'].append(loss)
        team_dict[team]['draws'].append(draw)

        team_dict[team]['wins_before'].append(0)
        team_dict[team]['losses_before'].append(0)
        team_dict[team]['draws_before'].append(0)
        
    else:
        team_dict[team]['wins_before'].append(team_dict[team]['wins'][-1])
        team_dict[team]['losses_before'].append(team_dict[team]['losses'][-1])
        team_dict[team]['draws_before'].append(team_dict[team]['draws'][-1])

        team_dict[team]['wins'].append(team_dict[team]['wins'][-1] + win)
        team_dict[team]['losses'].append(team_dict[team]['losses'][-1] + loss)
        team_dict[team]['draws'].append(team_dict[team]['draws'][-1] + draw)
    
    return team_dict



def add_record(games_df: pd.DataFrame, teams_dict: dict) -> pd.DataFrame:
    """

    Args:
        games_df (pd.DataFrame): A dataframe of games data
        teams_dict (dict): A dictionary of team data

    Returns:
        pd.DataFrame: A dataframe of games data with a record column
    """
    record_df = pd.DataFrame()

    for team, items in teams_dict.items():
        temp_df = pd.DataFrame(items)
        record_df = pd.concat([record_df, temp_df])

    record_df = record_df[['season', 'week', 'team', 'wins_before', 'losses_before', 'draws_before']]

    games_df = games_df.merge(record_df, 
                              left_on=['season', 'week', 'home_team'], 
                              right_on=['season', 'week', 'team'])
    games_df = games_df.merge(record_df, 
                             left_on=['season', 'week', 'away_team'], 
                             right_on=['season', 'week', 'team'],
                             suffixes=('', '_away'))

    return games_df
