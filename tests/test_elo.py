import pandas as pd
import numpy as np
from nfl_predict.elo import elo_test_score, elo_metric, elo_rate, evaluate_elo, simulate_game_result

def test_elo_test_score():
    # Test cases for elo_test_score
    assert elo_test_score({'elo_rating': 1500, 'elo_rating_away': 1450, 'result': 1}, 100) == 1
    assert elo_test_score({'elo_rating': 1500, 'elo_rating_away': 1450, 'result': -1}, 100) == 0
    assert elo_test_score({'elo_rating': 1400, 'elo_rating_away': 1450, 'result': -1}, 100) == 1

def test_elo_metric():
    # Sample data for testing
    df_val = pd.DataFrame({
        'home_team': ['Team A', 'Team B'],
        'season': [2023, 2023]
    })
    elos_df = pd.DataFrame({
        'team': ['Team A', 'Team B'],
        'season': [2023, 2023],
        'elo_rating': [1500, 1450]
    })
    result = elo_metric(df_val, elos_df, 100)
    assert not result.empty

def test_elo_rate():
    # Sample data for testing
    df_games = pd.DataFrame({
        'home_team': ['Team A'],
        'away_team': ['Team B'],
        'season': [2023]
    })
    teams = np.array(['Team A', 'Team B'])
    result_train, result_val = elo_rate(df_games, teams)
    assert isinstance(result_train, pd.DataFrame)
    assert isinstance(result_val, pd.DataFrame)

def test_evaluate_elo():
    # Mock data for testing
    df_training = pd.DataFrame({'data': [1]})
    df_val = pd.DataFrame({'data': [1]})
    teams = np.array(['Team A', 'Team B'])
    result = evaluate_elo(df_training, df_val, teams, 20, 100)
    assert result is not None

def test_simulate_game_result():
    # Sample data for testing
    elos_df = pd.DataFrame({
        'team': ['Team A', 'Team B'],
        'elo_rating': [1500, 1450]
    })
    probability = simulate_game_result('Team A', 'Team B', 100, elos_df)
    assert 0 <= probability <= 1
