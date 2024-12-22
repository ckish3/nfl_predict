import pandas as pd
import numpy as np
from math import isclose
from nfl_predict.evaluate import split_data, evaluate_moneyline


def test_split_data():
    # Create a sample DataFrame
    data = {
        'season': [2023] * 12 + [2024] * 8,
        'week': list(range(1, 13)) + list(range(1, 9)),
        'data': range(20)
    }
    data_df = pd.DataFrame(data)

    # Split the data
    df_train, df_val, df_test = split_data(data_df, training_week_cutoff=12, proportion_val=0.5)

    # Test that the training set contains only weeks 1-12
    assert all(df_train['week'] <= 12)
    assert all(df_test['week'] > 12
    )
    # Test that the validation and test sets are properly proportioned
    total_val_test = len(df_val) + len(df_test)
    assert total_val_test > 0  # Ensure there are validation and test samples
    assert isclose(len(df_val) / total_val_test, 0.5, rel_tol=0.1)  # Check proportion

def test_split_data_edge_case():
    # Create a DataFrame with insufficient data
    data = {
        'season': [2023],
        'week': [1],
        'data': [1]
    }
    data_df = pd.DataFrame(data)

    # Split the data
    df_train, df_val, df_test = split_data(data_df)

    # Check that training set is empty
    assert df_train.empty
    assert df_val.empty
    assert df_test.empty


def test_evaluate_spread():
    # Create a sample DataFrame for testing
    data = {
        'spread_line': [-3, 3, -5, 5],
        'result': [-1, 1, -1, 1]
    }
    df_test = pd.DataFrame(data)

    # Test the evaluate_spread function
    accuracy = evaluate_spread(df_test)
    assert accuracy == 1.0  # All predictions should match


def test_evaluate_spread_edge_case():
    # Create an empty DataFrame
    df_empty = pd.DataFrame(columns=['spread_line', 'result'])

    # Test the evaluate_spread function with an empty DataFrame
    accuracy = evaluate_spread(df_empty)
    assert accuracy == 0.0  # No games should result in 0 accuracy


def test_evaluate_moneyline():
    # Create a sample DataFrame for testing
    data = {
        'home_moneyline': [-150, 200, -100, 100],
        'result': [1, -1, 1, -1]
    }
    df_test = pd.DataFrame(data)

    # Test the evaluate_moneyline function
    accuracy = evaluate_moneyline(df_test)
    assert accuracy == 1.0  # All predictions should match

def test_evaluate_moneyline_edge_case():
    # Create an empty DataFrame
    df_empty = pd.DataFrame(columns=['home_moneyline', 'result'])

    # Test the evaluate_moneyline function with an empty DataFrame
    accuracy = evaluate_moneyline(df_empty)
    assert accuracy == 0.0  # No games should result in 0 accuracy

def test_evaluate_moneyline_with_nulls():
    # Create a DataFrame with null values
    data = {
        'home_moneyline': [None, -150, 200],
        'result': [1, -1, 1]
    }
    df_test = pd.DataFrame(data)

    # Test the evaluate_moneyline function
    accuracy = evaluate_moneyline(df_test)
    assert accuracy == 0.5  # Only one valid prediction should match


    def test_evaluate_home_advantage():
    # Create a sample DataFrame for testing
    data = {
        'location': ['Home', 'Home', 'Neutral', 'Away'],
        'result': [1, 1, 1, -1]
    }
    df_test = pd.DataFrame(data)

    # Test the evaluate_home_advantage function
    accuracy = evaluate_home_advantage(df_test)
    assert accuracy == 0.5  # Two home wins out of four games

def test_evaluate_home_advantage_edge_case():
    # Create an empty DataFrame
    df_empty = pd.DataFrame(columns=['location', 'result'])

    # Test the evaluate_home_advantage function with an empty DataFrame
    accuracy = evaluate_home_advantage(df_empty)
    assert accuracy == 0.0  # No games should result in 0 accuracy

def test_evaluate_home_advantage_with_neutral():
    # Create a DataFrame with neutral locations
    data = {
        'location': ['Neutral', 'Home', 'Home', 'Neutral'],
        'result': [1, 1, -1, -1]
    }
    df_test = pd.DataFrame(data)

    # Test the evaluate_home_advantage function
    accuracy = evaluate_home_advantage(df_test)
    assert accuracy == 0.5  # Only two valid home games should be counted