"""
Module for combining all steps in creating a cricket rating system

"""

__date__ = "2023-06-17"
__author__ = "SamKemp"


# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
from cricket_preprocessing import combine_pre_processing, filter_dataframe_by_date
from cricket_feature_engineering import create_bat_bowl_dfs
from cricket_rating_algorithm import get_full_ratings_df, merge_create_and_save_files, get_player_rating_ranking

# %% --------------------------------------------------------------------------
# Load dataset
# -----------------------------------------------------------------------------

df = pd.read_csv(r'..\datasets\formatted_bbb_df.csv')

# %% --------------------------------------------------------------------------
# (Optional) Choose subset date range of data
# -----------------------------------------------------------------------------

df = filter_dataframe_by_date(df, lower_date='2017-01-01', upper_date='2021-01-01')

# %% --------------------------------------------------------------------------
# Clean and pre-process data
# -----------------------------------------------------------------------------

clean_df = combine_pre_processing(df)

# %% --------------------------------------------------------------------------
# Create batting and bowling dataframes
# -----------------------------------------------------------------------------

bat_df, bowl_df = create_bat_bowl_dfs(clean_df)

# %% --------------------------------------------------------------------------
# Create full rating dataset
# -----------------------------------------------------------------------------

full_dataset = get_full_ratings_df(bat_df, bowl_df)

# %% --------------------------------------------------------------------------
# Create and save ranking files 
# -----------------------------------------------------------------------------

rank_df_batting, rank_df_bowling, rank_df_all_rounder = merge_create_and_save_files(full_dataset)

# %% --------------------------------------------------------------------------
# Find ranking and rating of a player
# -----------------------------------------------------------------------------

get_player_rating_ranking(full_dataset, 'Chris Gayle', 'batting', 2021, 1)
# %%
