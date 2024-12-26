"""
This module contains functions for computing the trueskill ratings
of NFL teams, and then using those ratings to predict game results
"""


from operator import ge
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import json
import trueskill
from transformers import tool


DF_ALL_TRUESKILLS = None # The dataframe of all trueskill ratings. It would be much better if this wasn't
    # a global variable, but the HuggingFace agent library doesn't allow tool inputs to be dataframes, 
    # so it has to be a global variable

def get_latest_ratings(df_all_trueskills: pd.DataFrame) -> dict:
    """
    Returns a dictionary of the latest trueskill ratings (the values for the latest week 
    in the latest season)

    Args:
        df_all_trueskills (pd.DataFrame): The dataframe of all trueskill ratings

    Returns:
        dict: the latest trueskill ratings for just the latest season
    """

    if df_all_trueskills is None:
        raise Exception('Ratings have not been calculated yet')

    latest_ratings = {}

    latest_season = max(df_all_trueskills['season'].values)
    season_df = df_all_trueskills[df_all_trueskills['season'] == latest_season]

    # You loop through each team in the latest season instead of just getting the maximum week
    # overall ecause teams may have a bye week in that latet week and therefore not appear there.
    # This gets the latest week for each team, which could be different weeks because of byes
    for team, team_df in season_df.groupby('team'):
        latest_week = max(team_df['week'].values)
        latest_df = team_df[team_df['week'] == latest_week]

        latest_ratings[team] = {'mu': latest_df['trueskill_rating'].values[0], 
                                'sigma': latest_df['trueskill_sigma'].values[0],}

    return latest_ratings


def trueskill_test_score(row, home_field_advantage):
    if row['trueskill_rating'] + home_field_advantage > row['trueskill_rating_away'] and row['result'] > 0:
        return 1

    if row['trueskill_rating'] + home_field_advantage < row['trueskill_rating_away'] and row['result'] < 0:
        return 1

    return 0


def trueskill_metric(df_val, trueskills_df, home_field_advantage):
    df_val = df_val.merge(trueskills_df, 
                            left_on=['home_team', 'season'],
                            right_on=['team', 'season'])
    df_val = df_val.merge(trueskills_df, 
                            left_on=['away_team', 'season'],
                            right_on=['team', 'season'],
                            suffixes=('', '_away'))
    df_val['trueskill_test'] = df_val.apply(trueskill_test_score, home_field_advantage=home_field_advantage, axis=1)
    
    return df_val[['trueskill_test']].mean()['trueskill_test']


def trueskill_rate(df_games: pd.DataFrame, teams: np.ndarray, home_field_advantage: int=100, beta: float=4.17, tau: float=0.083) -> None:

    global DF_ALL_TRUESKILLS

    team_final_trueskills = {
            'season': [],
            'team': [],
            'trueskill_rating': [],
            'trueskill_sigma': []
        }

    all_trueskills = {
            'season': [],
            'week': [],
            'team': [],
            'trueskill_rating': [],
            'trueskill_sigma': []
        }

    for season, df_season in df_games.groupby('season'):
        
        df_season = df_season.sort_values('week', ascending=True)
        season_trueskill_ratings = {}

        
        for team in teams:
            season_trueskill_ratings[team] = trueskill.Rating()
        
        for i in df_season.index:
            row = df_season.loc[i]
            if row['result'] > 0:
                is_draw = False
                winner = row['home_team']
                loser = row['away_team']
                r_winner_before = trueskill.Rating(mu=season_trueskill_ratings[winner].mu + home_field_advantage,
                                                   sigma=season_trueskill_ratings[winner].sigma)
                r_loser_before = season_trueskill_ratings[loser]
            
            elif row['result'] < 0:
                is_draw = False
                winner = row['away_team']
                loser = row['home_team']

                r_winner_before = season_trueskill_ratings[winner]
                r_loser_before = trueskill.Rating(mu=season_trueskill_ratings[loser].mu + home_field_advantage,
                                                  sigma=season_trueskill_ratings[loser].sigma)
            else:
                is_draw = True

                winner = row['home_team'] # The choice of "winner" and "loser" is now arbitrary
                loser = row['away_team']
                r_winner_before = trueskill.Rating(mu=season_trueskill_ratings[winner].mu + home_field_advantage,
                                                   sigma=season_trueskill_ratings[winner].sigma)
                r_loser_before = season_trueskill_ratings[loser]
            
                
            r_winner, r_loser = trueskill.rate_1vs1(r_winner_before, 
                                                    r_loser_before,
                                                    drawn=is_draw, 
                                                    env=trueskill.TrueSkill(backend='mpmath', beta=beta, tau=tau)
)
            if row['result'] >= 0:
                season_trueskill_ratings[winner] = trueskill.Rating(mu=season_trueskill_ratings[winner].mu + (r_winner.mu - r_winner_before.mu),
                                                                    sigma=r_winner.sigma)          
                season_trueskill_ratings[loser] = r_loser
            elif row['result'] < 0:
                season_trueskill_ratings[winner] = r_winner
                season_trueskill_ratings[loser] = trueskill.Rating(mu=season_trueskill_ratings[loser].mu + (r_loser.mu - r_loser_before.mu),
                                                                    sigma=r_loser.sigma)
                        
            all_trueskills['season'].append(row['season'])
            all_trueskills['week'].append(row['week'])
            all_trueskills['team'].append(winner)
            all_trueskills['trueskill_rating'].append(season_trueskill_ratings[winner].mu)
            all_trueskills['trueskill_sigma'].append(season_trueskill_ratings[winner].sigma)

            all_trueskills['season'].append(row['season'])
            all_trueskills['week'].append(row['week'])
            all_trueskills['team'].append(loser)
            all_trueskills['trueskill_rating'].append(season_trueskill_ratings[loser].mu)
            all_trueskills['trueskill_sigma'].append(season_trueskill_ratings[loser].sigma)

        for team in teams:
            team_final_trueskills['team'].append(team)
            team_final_trueskills['season'].append(season)
            team_final_trueskills['trueskill_rating'].append(season_trueskill_ratings[team].mu)
            team_final_trueskills['trueskill_sigma'].append(season_trueskill_ratings[team].sigma)
            

    DF_ALL_TRUESKILLS = pd.DataFrame(all_trueskills)

    trueskills_df = pd.DataFrame(team_final_trueskills)

    return trueskills_df


def evaluate_trueskill(df_training, df_val, teams, beta, home_field_advantage, tau):
    global DF_ALL_TRUESKILLS
    trueskills_df = trueskill_rate(df_training, teams, home_field_advantage=home_field_advantage, beta=beta, tau=tau)
    return trueskill_metric(df_val, trueskills_df, home_field_advantage)

@tool
def predict_game_result(home_team: str, away_team: str, home_field_advantage: int, beta: float) -> float:
    """
    Returns the probability that the home team will win the game against the away team given the home field advantage that
    the home team has and trueskill ratings.
    
    Args:
        home_team: (str) The home team in the game
        away_team: (str) The away team in the game
        home_field_advantage: (int) The home field advantage for the Trueskill rating
        trueskill_df: (pd.DataFrame) A dataframe of trueskill ratings for just the latest season
        beta: (float) The beta parameter for the Trueskill rating

    Returns:
        float: The probability that the home team will win the game
    """

    latest_ratings = get_latest_ratings(DF_ALL_TRUESKILLS)
    mu_combined = latest_ratings[home_team]['mu'] + home_field_advantage - latest_ratings[away_team]['mu']

    sigma = math.sqrt(latest_ratings[home_team]['sigma']**2 + latest_ratings[away_team]['sigma']**2)
    denom = math.sqrt(2 * (beta * beta) + sigma)
    ts = trueskill.global_env()
    return ts.cdf(mu_combined / denom)


@tool
def plot_team_ratings(team: str, season: int) -> str:
    """
    Plots the ratings of a tem throughout the season given by season. 
    
    Args:
        team: (str) The team whose ratings to plot
        season: (int) The season of ratings to plot

    Returns:
        str: a list of the team's ratings converted into a string
    """
    global DF_ALL_TRUESKILLS
    season_df = DF_ALL_TRUESKILLS[(DF_ALL_TRUESKILLS['team'] == team) & (DF_ALL_TRUESKILLS['season'] == season)]
    season_df = season_df.sort_values('week', ascending=True)
    x = season_df['week'].values
    y = season_df['trueskill_rating'].values
    plt.plot(x, y)

    plt.xlabel('Week')
    plt.ylabel('Trueskill Rating')
    plt.title(f'Trueskill Ratings for {team} in Season {season}')
    plt.show()

    return json.dumps(y.tolist())
