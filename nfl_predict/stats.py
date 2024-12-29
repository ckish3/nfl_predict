"""
This module contains functions for downloading and computing weekly team stats
"""

import pandas as pd
import datetime
import json

from transformers import tool

import nfl_data_py as nfl


WEEKLY_TEAM_STATS = None # Ideally this would not be a global variable, 
#but the HuggingFace agent library doesn't allow tool inputs to be dictionaries


def download_weekly_player_stats() -> pd.DataFrame:
    """
    Downloads weekly player stats from nfl_data_py

    Returns:
        pd.DataFrame: A dataframe of weekly player stats
    """
    
    current_year = datetime.datetime.now().year
    years = [x for x in range(1999, current_year + 1)]

    player_stats = nfl.import_weekly_data(years)

    return player_stats


def calculate_weekly_team_stats() -> None:
    """
    Computes weekly team stats from weekly player stats
    """
    
    global WEEKLY_TEAM_STATS
    columns = ['completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions',
               'sacks', 'sack_yards', 'sack_fumbles', 'sack_fumbles_lost', 'passing_air_yards',
                'passing_yards_after_catch', 'passing_first_downs',
                'passing_2pt_conversions', 'carries', 'rushing_yards',
                'rushing_tds', 'rushing_fumbles', 'rushing_fumbles_lost',
                'rushing_first_downs', 'rushing_2pt_conversions',
                'receptions', 'targets', 'receiving_yards', 'receiving_tds',
                'receiving_fumbles', 'receiving_fumbles_lost', 'receiving_air_yards',
                'receiving_yards_after_catch', 'receiving_first_downs',
                'receiving_2pt_conversions',
                'special_teams_tds', 'fantasy_points', 'fantasy_points_ppr']

                #columnw where sum is not appropriate: 'passing_epa', 'receiving_epa', 'rushing_epa',
                # 'racr', 'target_share', 'air_yards_share', 'wopr', 'pacr', 'dakota', 
    player_df = download_weekly_player_stats()
    
    team_dict = player_df.groupby(['recent_team', 'season', 'week'])[columns].sum().to_dict(orient='index')

    WEEKLY_TEAM_STATS = team_dict

@tool
def get_weekly_team_stats(team: str, season: int, week: int) -> str:
    """
    Returns theteam stats for a given team, season, and week

    Args:
        team: (str) The team to get stats for
        season: (int) The season to get stats for
        week: (int) The week to get stats for

    Returns:
        str: A string of the team stats for the given week
    """

    if WEEKLY_TEAM_STATS is None:
        calculate_weekly_team_stats()
    
    stats = WEEKLY_TEAM_STATS[(team, season, week)]
    stats_str = json.dumps(stats)
    print(stats_str)
    return stats_str
