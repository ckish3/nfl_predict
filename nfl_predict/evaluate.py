
"""
This module contains functions for evaluating various methods to predict the
result of NFL games

"""
import math
import pandas as pd
import random
from typing import Tuple

def split_data(data_df: pd.DataFrame, training_week_cutoff: int=12, proportion_val: float=0.5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into training, validation, and test dataframes. The training set is all  the games
    in week 1 through training_week_cutoff in all seasons. The validation and test sets are a random sample 
    of the games in all seasons after training_week_cutoff. 

    Args:
        data_df (pd.DataFrame): A dataframe of games data
        training_week_cutoff (int, optional): The week to stop training on. Defaults to 12.
        proportion_val (float, optional): The proportion of data to use for validation. Defaults to 0.5.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple of dataframes for training, validation, and test data
    """


    def test_val_definer(week):
        if week <= training_week_cutoff:
            return 'training'
        if random.random() < proportion_val:
            return 'validation'
        else:
            return 'test'

    df_def = data_df[['season', 'week']]
    df_def = df_def.groupby(['season', 'week']).count().reset_index()

    df_def['data_split'] = df_def['week'].apply(test_val_definer)

    data_df = data_df.merge(df_def, on=['season', 'week'])
    
    df_training = data_df[data_df['data_split'] == 'training']
    df_val = data_df[data_df['data_split'] == 'validation']
    df_test = data_df[data_df['data_split'] == 'test']

    return df_training, df_val, df_test


def evaluate_spread(df_test: pd.DataFrame) -> float:
    """
    Args:
        df_test (pd.DataFrame): A dataframe of games data

    Returns:
        float: The accuracy of the spread at predicting a win/lost
    """

    def spread_match2(row):
        if (row['spread_line'] < 0 and row['result'] < 0) or \
            (row['spread_line'] > 0 and row['result'] > 0):
            return 1
        else:
            return 0

    df_test['spread_test'] = df_test.apply(spread_match2, axis=1)

    return df_test['spread_test'].mean()

def evaluate_moneyline(df_test: pd.DataFrame) -> float:
    """
    Args:
        df_test (pd.DataFrame): A dataframe of games data

    Returns:
        float: The accuracy of the moneyline at predicting a win/lost
    """

    def moneyline_match2(row):
        if pd.isnull(row['home_moneyline']):
            return None
        if (row['home_moneyline'] < 0 and row['result'] > 0) or \
            (row['home_moneyline'] > 0 and row['result'] < 0):
            return 1
        else:
            return 0


    df_test['moneyline_test'] = df_test.apply(moneyline_match2, axis=1)
    
    return df_test['moneyline_test'].mean()


def evaluate_home_advantage(df_test: pd.DataFrame) -> float:
    """
    Args:
        df_test (pd.DataFrame): A dataframe of games data

    Returns:
        float: The accuracy of playing at home at predicting a win/lost
    """

    def location_test2(row):
        if (row['result'] > 0 and row['location'] != 'Neutral'):
            return 1
        else:
            return 0
            
    df_test['home_test'] = df_test.apply(location_test2, axis=1) 
    return df_test['home_test'].mean()

def evaluate_record(df_test: pd.DataFrame) -> float:
    """
    Args:
        df_test (pd.DataFrame): A dataframe of games data

    Returns:
        float: The accuracy of having a better record at predicting a win/lost
    """

    def record_test(row):
        win_percentage = row['wins_before'] / (row['wins_before'] + row['losses_before'] + row['draws_before'] + .00001)
        win_percentage_away = row['wins_before_away'] / (row['wins_before_away'] + row['losses_before_away'] + row['draws_before_away'] + .00001)

        if win_percentage > win_percentage_away:
            record_result = 1
        elif win_percentage < win_percentage_away:
            record_result = -1
        elif row['location'] == 'Home':
            record_result = 1
        else:
            record_result = -1

        if record_result == math.copysign(1, row['result']):
            return 1
        else:
            return 0

    df_test['record_test'] = df_test.apply(record_test, axis=1)    
    return df_test['record_test'].mean()