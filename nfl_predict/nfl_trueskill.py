

from typing import Tuple
import pandas as pd
import numpy as np
import scipy.stats as stats
import trueskill


def trueskill_test_score(row, home_field_advantage):
    if row['trueskill_rating'] + home_field_advantage > row['trueskill_rating_away'] and row['result'] > 0:
        return 1

    if row['trueskill_rating'] + home_field_advantage < row['trueskill_rating_away'] and row['result'] < 0:
        return 1

    return 0


def trueskill_metric(df_val, elos_df, home_field_advantage):
    df_val = df_val.merge(elos_df, 
                            left_on=['home_team', 'season'],
                            right_on=['team', 'season'])
    df_val = df_val.merge(elos_df, 
                            left_on=['away_team', 'season'],
                            right_on=['team', 'season'],
                            suffixes=('', '_away'))

    df_val['trueskill_test'] = df_val.apply(trueskill_test_score, home_field_advantage=home_field_advantage, axis=1)
    
    return df_val[['trueskill_test']].mean()['trueskill_test']


def trueskill_rate(df_games: pd.DataFrame, teams: np.ndarray, home_field_advantage: int=100, beta: float=4.17, tau: float=0.083) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    
    eloLeague = trueskill.Elo(k=0)
    
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
                        
            all_trueskills['season'].append(row['season')
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

    df_all_trueskills = pd.DataFrame(all_trueskills)

    trueskills_df = pd.DataFrame(team_final_trueskills)

    return df_all_trueskills, trueskills_df

def evaluate_trueskill(df_training, df_val, teams, beta, home_field_advantage, tau):
    df_all_trueskills, trueskills_df = trueskill_rate(df_training, teams, home_field_advantage=home_field_advantage, beta=beta, tau=tau)
    return trueskill_metric(df_val, trueskills_df, home_field_advantage)


def simulate_game_result(home_team: str, away_team: str, home_field_advantage: int, trueskill_df: pd.DataFrame) -> float:
    """
    Returns the probability that the home team will win the game against the away team given the home field advantage that
    the home team has and ELO ratings given by the pandas dataframe elos_df.
    
    Args:
        home_team (str): The home team in the game
        away_team (str): The away team in the game
        home_field_advantage (int): The home field advantage for the Elo rating
        trueskill_df (pd.DataFrame): A dataframe of trueskill ratings for just the latest season

    Returns:
        float: The probability that the home team will win the game
    """

    mu_combined = trueskill_df[trueskill_df['team'] == home_team]['trueskill_rating'].values[0] + home_field_advantage - \
        trueskill_df[trueskill_df['team'] == away_team]['trueskill_rating'].values[0]

    sigma = sqrt(trueskill_df[trueskill_df['team'] == home_team]['trueskill_sigma'].values[0]**2 + trueskill_df[trueskill_df['team'] == away_team]['trueskill_sigma'].values[0]**2)

    return 1 - stats.norm.cdf(0, loc=mu_combined, scale=sigma)
    