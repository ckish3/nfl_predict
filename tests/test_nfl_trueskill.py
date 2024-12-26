import pandas as pd
from nfl_predict.nfl_trueskill import get_latest_ratings, trueskill_test_score, trueskill_metric, trueskill_rate, evaluate_trueskill, plot_team_ratings

import unittest
from unittest.mock import patch
import json


def test_get_latest_ratings():
    # Create a sample DataFrame for testing
    data = {
        'season': [2023, 2023, 2023, 2024, 2024],
        'week': [1, 1, 2, 1, 2],
        'team': ['Team A', 'Team B', 'Team A', 'Team A', 'Team B'],
        'trueskill_rating': [25.0, 30.0, 26.0, 27.0, 31.0],
        'trueskill_sigma': [8.0, 7.0, 7.5, 8.5, 6.5]
    }
    df_all_trueskills = pd.DataFrame(data)

    # Test the get_latest_ratings function
    latest_ratings = get_latest_ratings(df_all_trueskills)
    assert latest_ratings['Team A'] == {'mu': 27.0, 'sigma': 8.5}  # Latest rating for Team A
    assert latest_ratings['Team B'] == {'mu': 31.0, 'sigma': 6.5}  # Latest rating for Team B

def test_get_latest_ratings_empty():
    # Create an empty DataFrame
    df_empty = pd.DataFrame(columns=['season', 'week', 'team', 'trueskill_rating', 'trueskill_sigma'])

    # Test the get_latest_ratings function with an empty DataFrame
    latest_ratings = get_latest_ratings(df_empty)
    assert latest_ratings == {}  # Should return an empty dictionary

def test_get_latest_ratings_none():
    # Test the get_latest_ratings function with None
    try:
        get_latest_ratings(None)
    except Exception as e:
        assert str(e) == 'Ratings have not been calculated yet'  # Should raise an exception


def test_trueskill_test_score():
    # Test cases for trueskill_test_score
    row1 = {'trueskill_rating': 25.0, 'trueskill_rating_away': 20.0, 'result': 1}
    assert trueskill_test_score(row1, 5) == 1  # Home team wins

    row2 = {'trueskill_rating': 20.0, 'trueskill_rating_away': 25.0, 'result': -1}
    assert trueskill_test_score(row2, -5) == 1  # Away team wins

    row3 = {'trueskill_rating': 30.0, 'trueskill_rating_away': 30.0, 'result': 1}
    assert trueskill_test_score(row3, 0) == 0  # Draw, not a win

    row4 = {'trueskill_rating': 15.0, 'trueskill_rating_away': 20.0, 'result': -1}
    assert trueskill_test_score(row4, 10) == 0  # Home team should not win

def test_trueskill_test_score_edge_case():
    # Test case with missing data
    row = {'trueskill_rating': None, 'trueskill_rating_away': None, 'result': 1}
    assert trueskill_test_score(row, 0) == 0  # Should handle None gracefully


def test_trueskill_metric():
    # Create sample DataFrames for testing
    df_val = pd.DataFrame({
        'home_team': ['Team A', 'Team B'],
        'away_team': ['Team C', 'Team D'],
        'season': [2023, 2023]
    })

    trueskills_df = pd.DataFrame({
        'team': ['Team A', 'Team B', 'Team C', 'Team D'],
        'season': [2023, 2023, 2023, 2023],
        'trueskill_rating': [25.0, 30.0, 20.0, 15.0],
        'trueskill_sigma': [8.0, 7.0, 6.0, 5.0]
    })

    # Test the trueskill_metric function
    average_accuracy = trueskill_metric(df_val, trueskills_df, home_field_advantage=5)
    assert average_accuracy == 1.0  # Adjust based on expected output

def test_trueskill_metric_empty():
    # Create empty DataFrames
    df_val_empty = pd.DataFrame(columns=['home_team', 'away_team', 'season'])
    trueskills_df_empty = pd.DataFrame(columns=['team', 'season', 'trueskill_rating', 'trueskill_sigma'])

    # Test the trueskill_metric function with empty DataFrames
    average_accuracy = trueskill_metric(df_val_empty, trueskills_df_empty, home_field_advantage=5)
    assert average_accuracy == 0.0  # Should return 0 for empty input

def test_trueskill_rate():
    # Create a sample DataFrame for testing
    df_games = pd.DataFrame({
        'home_team': ['Team A', 'Team B', 'Team A'],
        'away_team': ['Team B', 'Team A', 'Team C'],
        'season': [2023, 2023, 2023],
        'result': [1, -1, 1]  # Team A wins first and last game, Team B wins the second
    })

    teams = np.array(['Team A', 'Team B', 'Team C'])

    # Test the trueskill_rate function
    trueskills_df = trueskill_rate(df_games, teams, home_field_advantage=5)
    assert not trueskills_df.empty  # Ensure the DataFrame is not empty
    assert set(trueskills_df['team']) == set(teams)  # Ensure all teams are present

def test_trueskill_rate_empty():
    # Create an empty DataFrame
    df_games_empty = pd.DataFrame(columns=['home_team', 'away_team', 'season', 'result'])
    teams = np.array(['Team A', 'Team B'])

    # Test the trueskill_rate function with an empty DataFrame
    trueskills_df = trueskill_rate(df_games_empty, teams, home_field_advantage=5)
    assert trueskills_df.empty  # Should return an empty DataFrame


def test_evaluate_trueskill():
    # Create sample DataFrames for training and validation
    df_training = pd.DataFrame({
        'home_team': ['Team A', 'Team B'],
        'away_team': ['Team B', 'Team A'],
        'season': [2023, 2023],
        'result': [1, -1]  # Team A wins the first game, Team B wins the second
    })

    df_val = pd.DataFrame({
        'home_team': ['Team A', 'Team C'],
        'away_team': ['Team C', 'Team B'],
        'season': [2023, 2023]
    })

    teams = np.array(['Team A', 'Team B', 'Team C'])

    # Test the evaluate_trueskill function
    accuracy = evaluate_trueskill(df_training, df_val, teams, beta=4.17, home_field_advantage=5, tau=0.083)
    assert accuracy >= 0  # Ensure the accuracy is a valid number

def test_evaluate_trueskill_empty():
    # Create empty DataFrames
    df_training_empty = pd.DataFrame(columns=['home_team', 'away_team', 'season', 'result'])
    df_val_empty = pd.DataFrame(columns=['home_team', 'away_team', 'season'])
    teams = np.array(['Team A', 'Team B'])

    # Test the evaluate_trueskill function with empty DataFrames
    accuracy = evaluate_trueskill(df_training_empty, df_val_empty, teams, beta=4.17, home_field_advantage=5, tau=0.083)
    assert accuracy == 0.0  # Should return 0 for empty input

def test_predict_game_result():
    # Create a sample DataFrame for TrueSkill ratings
    data = {
        'season': [2023, 2023],
        'team': ['Team A', 'Team B'],
        'trueskill_rating': [25.0, 30.0],
        'trueskill_sigma': [8.0, 7.0]
    }
    global DF_ALL_TRUESKILLS
    DF_ALL_TRUESKILLS = pd.DataFrame(data)

    # Test the predict_game_result function
    probability = predict_game_result('Team A', 'Team B', home_field_advantage=5, beta=4.17)
    assert 0 <= probability <= 1  # Probability should be between 0 and 1

def test_predict_game_result_team_not_found():
    # Create a sample DataFrame for TrueSkill ratings
    data = {
        'season': [2023],
        'team': ['Team A'],
        'trueskill_rating': [25.0],
        'trueskill_sigma': [8.0]
    }
    global DF_ALL_TRUESKILLS
    DF_ALL_TRUESKILLS = pd.DataFrame(data)

    # Test the predict_game_result function with a team not in the ratings
    try:
        predict_game_result('Team A', 'Team B', home_field_advantage=5, beta=4.17)
    except KeyError as e:
        assert str(e) == "'Team B'"  # Should raise an exception for missing team


# Assuming the function is in the nfl_trueskill module
from nfl_predict.nfl_trueskill import plot_team_ratings

class TestPlotTeamRatings(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.sample_data = {
            'team': ['Team A', 'Team A', 'Team B', 'Team A'],
            'season': [2023, 2023, 2023, 2023],
            'week': [1, 2, 1, 3],
            'trueskill_rating': [1500, 1520, 1480, 1530]
        }
        global DF_ALL_TRUESKILLS
        DF_ALL_TRUESKILLS = pd.DataFrame(self.sample_data)

    @patch('nfl_predict.nfl_trueskill.plt.show')  # Mock plt.show
    @patch('nfl_predict.nfl_trueskill.plt.plot')  # Mock plt.plot
    def test_plot_team_ratings(self, mock_plot, mock_show):
        result = plot_team_ratings('Team A', 2023)
        
        # Check if the plot function was called
        mock_plot.assert_called_once()
        
        # Check the return value
        expected_output = json.dumps([1500, 1520, 1530])
        self.assertEqual(result, expected_output)

    def test_invalid_team(self):
        with self.assertRaises(KeyError):
            plot_team_ratings('Invalid Team', 2023)

    def test_invalid_season(self):
        with self.assertRaises(KeyError):
            plot_team_ratings('Team A', 2024)

if __name__ == '__main__':
    unittest.main()