
import json
import stat
import matplotlib.pyplot as plt
import os
from huggingface_hub import login, InferenceClient
from transformers import CodeAgent, HfApiEngine, ReactCodeAgent, tool

from nfl_trueskill import predict_game_result, plot_team_ratings
from stats import get_weekly_team_stats
import nfl_trueskill
import stats 
import prepare_game_data

def main():

    login(os.environ['HF_WRITE_TOKEN'])

    llm_engine = HfApiEngine(model="Qwen/Qwen2.5-72B-Instruct")
    agent = ReactCodeAgent(tools=[nfl_trueskill.predict_game_result, 
                                  nfl_trueskill.plot_team_ratings, 
                                  stats.get_weekly_team_stats,
                                  ], 
                            llm_engine=llm_engine, 
                            additional_authorized_imports=['matplotlib', 'json'])


    full_df = prepare_game_data.download_data()
    games_df = prepare_game_data.get_historical_games(full_df)
    teams = prepare_game_data.get_all_teams(games_df)
    nfl_trueskill.trueskill_rate(games_df, teams, home_field_advantage=2.5, beta=4)

    stats.calculate_weekly_team_stats()

    prompts = [
        "Plot the ratings of CLE during the 2020 season",
        "Predict the result of the game CLE at NYJ with a home field advantage of 2.5 and a beta of 4",
        "Get the stats for NYJ in week 1 of the 2020 season",]

    for prompt in prompts:
        agent.run(prompt)


if __name__ == "__main__":
    main()