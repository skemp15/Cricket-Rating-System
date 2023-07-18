"""
This is a module for performing a FIFA Ultimate Time player card price prediction.
The main functions will take the json input from the POST request containing a page
number and a player name and return the predicted valuation.
"""

__date__ = "2023-05-03"
__author__ = "SamKemp"


# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import pandas as pd
import logging
import json
from cricket_preprocessing import combine_pre_processing, filter_dataframe_by_date
from cricket_feature_engineering import create_bat_bowl_dfs
from cricket_rating_algorithm import get_full_ratings_df, get_player_rating_ranking


# %% --------------------------------------------------------------------------
# Set up logger
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# %% --------------------------------------------------------------------------
# Define rating retriever function
# -----------------------------------------------------------------------------
def retrieve_rating(json_data):

    logger.info('Starting process')

    input_data_start = json_data['Data Start']
    if input_data_start == "None":
        input_data_start = None
    input_data_end = json_data['Data End']
    if input_data_start == "None":
        input_data_start = None
    input_name = json_data['Player Name']
    input_style = json_data['Play Style']
    input_year = json_data['Year']
    input_quarter = json_data['Quarter']

    # Load data
    logger.info('Loading data')
    df = pd.read_csv(r'formatted_bbb_df.csv')

    # Create subset of date range
    logger.info('Creating a subset of date range')
    df = filter_dataframe_by_date(df, lower_date=input_data_start, upper_date=input_data_end)

    # Clean and pre-process data
    logger.info('Cleaning data')
    clean_df = combine_pre_processing(df)

    # Create batting and bowling dataframes
    logger.info('Creating batting and bowling dataframes')
    bat_df, bowl_df = create_bat_bowl_dfs(clean_df)

    # Create full rating dataset
    logger.info('Creating full rating dataset')
    full_dataset = get_full_ratings_df(bat_df, bowl_df)

    # Find ranking and rating of a player
    logger.info('Find ranking and rating of a player')
    results_dict = get_player_rating_ranking(full_dataset, input_name, input_style, input_year, input_quarter)

    return json.dumps(results_dict)

# %%
