"""
Python module containing functions for creating a rating algorithm 

"""

__date__ = "2023-06-17"
__author__ = "SamKemp"


# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

# %% --------------------------------------------------------------------------
# Function for normalising ratings out of 1000
# -----------------------------------------------------------------------------

def normalise_rating_scale(df, rating):

    min_value = df[rating].min()
    max_value = df[rating].max() * 1.1

    df[rating] = (df[rating] - min_value) * (1000 / (max_value - min_value))

    return df

# %% --------------------------------------------------------------------------
# Function for creating overall ratings for batting and bowling dataframes
# -----------------------------------------------------------------------------

def create_overall_rating_df(bat_df, bowl_df):

    # Batters

    # Create a new column for the overall_rating_batting
    bat_df['overall_rating_batting'] = 0.0

    # Iterate over each unique batsman_striker_name
    for batsman in bat_df['batsman_striker_name'].unique():
        # Get the subset of rows for the current batsman
        subset = bat_df[bat_df['batsman_striker_name'] == batsman]

        # Initialise the cumulative sum variable
        cumulative_sum = 0.0

        # Iterate over each row in the subset
        for i in range(len(subset)):
            # Update the cumulative sum of the log of the games played
            cumulative_sum += np.log1p(subset.iloc[i]['games_played'])

            # Calculate the overall rating
            overall_rating = (subset.iloc[:i+1]['rating_score_normalised'] * # Normalised rating score (capped at 10)
                                subset.iloc[:i+1]['bowler_rating_normalised'] * # Normalised opponent rating (capped between 0.5 and 1.5)
                                np.log1p(subset.iloc[:i+1]['games_played']) * # Log of games played
                                subset.iloc[:i+1]['player_mean_normalised']).sum() / cumulative_sum ** 0.75 # We take the cumulative sum to the power of 0.75 to favour players who have played longer

            # Assign the weighted average to the corresponding row
            bat_df.at[subset.index[i], 'overall_rating_batting'] = overall_rating

    # Normalise rating scale
    bat_df = normalise_rating_scale(bat_df, 'overall_rating_batting')

    # Bowlers

    # Create a new column for the overall_rating_bowling
    bowl_df['overall_rating_bowling'] = 0.0

    # Iterate over each unique bowler_name
    for bowler in bowl_df['bowler_name'].unique():
        # Get the subset of rows for the current batsman
        subset = bowl_df[bowl_df['bowler_name'] == bowler]

        # Initialize the cumulative sum variable
        cumulative_sum = 0.0

        # Iterate over each row in the subset
        for i in range(len(subset)):
            # Update the cumulative sum of the log of the games played
            cumulative_sum += np.log1p(subset.iloc[i]['games_played'])

            # Calculate the overall rating
            overall_rating = (subset.iloc[:i+1]['rating_score_normalised'] * # Normalised rating score (capped at 10)
                                subset.iloc[:i+1]['batter_rating_normalised'] * # Normalised opponent rating (capped between 0.5 and 1.5)
                                np.log1p(subset.iloc[:i+1]['games_played'])  * # Log of games played
                                subset.iloc[:i+1]['player_mean_normalised'] ).sum() / cumulative_sum ** 0.75 # We take the cumulative sum to the power of 0.75 to favour players who have played longer

            # Assign the weighted average to the corresponding row
            bowl_df.at[subset.index[i], 'overall_rating_bowling'] = overall_rating

    # Normalise rating scale
    bowl_df = normalise_rating_scale(bowl_df, 'overall_rating_bowling')

    return bat_df, bowl_df

# %% --------------------------------------------------------------------------
# Function for creating full rating dataframe
# -----------------------------------------------------------------------------

def get_full_ratings_df(bat_df, bowl_df):

    bat_df2, bowl_df2 = create_overall_rating_df(bat_df, bowl_df)

    # We rename our player names to be consistent
    bat_df2 = bat_df2.rename({'batsman_striker_name': 'player_name'}, axis=1)
    bowl_df2 = bowl_df2.rename({'bowler_name': 'player_name'}, axis=1)

    # We merge our dataframes
    merge_df = pd.merge(bat_df2, bowl_df2, on=['player_name', 'event_id', 'date'], suffixes=('_bat', '_bowl'), how='outer')

    # Fill any nan values 
    merge_df = merge_df.fillna(0)

    # Create an all-rounder column
    merge_df['overall_rating_all_rounder'] = (merge_df['overall_rating_batting'] * merge_df['overall_rating_bowling']) / 1000

    # Create games played for all-rounders
    merge_df['games_played_all_round'] = merge_df.groupby('player_name').cumcount() + 1

    return merge_df

# %% --------------------------------------------------------------------------
# Function for getting monthly mean ratings and rankings 
# -----------------------------------------------------------------------------
import pandas as pd

def get_rankings(df, rating_type):
    rating_column = 'overall_rating_' + rating_type

    # Get month and year
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year

    # Create a dataframe with all possible combinations of 'player_name', 'month', and 'year'
    all_combinations = pd.MultiIndex.from_product([df['player_name'].unique(), df['quarter'].unique(), df['year'].unique()], 
                                                  names=['player_name', 'quarter', 'year'])
    all_combinations = all_combinations.to_frame(index=False)

    # Merge the original data with all combinations to fill missing values
    filled_data = pd.merge(all_combinations, df, how='left', on=['player_name', 'quarter', 'year'])

    # Forward fill missing values
    filled_data[rating_column].fillna(method='ffill', inplace=True)

    # Group the data by 'player_name', 'month', and 'year', and find the mean rating for each group
    grouped_data = filled_data.groupby(['player_name', 'quarter', 'year']).agg({rating_column: 'mean'}).reset_index()

    # Sort the data by 'year', 'month', and rating in descending, ascending, and descending order respectively
    sorted_data = grouped_data.sort_values(['year', 'quarter', rating_column], ascending=[False, False, False])

    # Reset the index of the sorted data and rename the columns
    sorted_data.reset_index(drop=True, inplace=True)
    sorted_data.columns = ['player_name', 'quarter', 'year', rating_column]

    # Assign the ranking based on the sorted data for each month
    sorted_data['ranking'] = sorted_data.groupby(['year', 'quarter'])[rating_column].rank(method='dense', ascending=False)

    return sorted_data



# %% --------------------------------------------------------------------------
# Create and save ranking files
# -----------------------------------------------------------------------------

def merge_create_and_save_files(df):

    # Create our ranking dataframes
    df_batting = get_rankings(df, 'batting')
    df_bowling = get_rankings(df, 'bowling')
    df_all_rounder = get_rankings(df, 'all_rounder')

    # Create  year and quarter columns on our dataframe
    df['quarter'] = pd.to_datetime(df['date']).dt.quarter
    df['year'] = pd.to_datetime(df['date']).dt.year

    # Find the max of each games played at each quarter
    max_games_played_bat = df.groupby(['player_name', 'year', 'quarter'])['games_played_bat'].max().reset_index()
    max_games_played_bowl = df.groupby(['player_name', 'year', 'quarter'])['games_played_bowl'].max().reset_index()
    max_games_played_all_round = df.groupby(['player_name', 'year', 'quarter'])['games_played_all_round'].max().reset_index()

    # Merge dataframes to get games played at each quarter
    rank_df_batting = pd.merge(df_batting, max_games_played_bat[['player_name', 'year', 'quarter', 'games_played_bat']], on=['player_name', 'year', 'quarter'], how='left') 
    rank_df_bowling = pd.merge(df_bowling, max_games_played_bowl[['player_name', 'year', 'quarter', 'games_played_bowl']], on=['player_name', 'year', 'quarter'], how='left') 
    rank_df_all_rounder = pd.merge(df_all_rounder, max_games_played_all_round[['player_name', 'year', 'quarter', 'games_played_all_round']], on=['player_name', 'year', 'quarter'], how='left') 

    # Back fill any missing values and fill the rest with 0
    rank_df_batting['games_played_bat'] = rank_df_batting.groupby('player_name')['games_played_bat'].fillna(method='bfill').fillna(1)
    rank_df_bowling['games_played_bowl'] = rank_df_bowling.groupby('player_name')['games_played_bowl'].fillna(method='bfill').fillna(1)
    rank_df_all_rounder['games_played_all_round'] = rank_df_all_rounder.groupby('player_name')['games_played_all_round'].fillna(method='bfill').fillna(1)

    # Save to files
    rank_df_batting.to_csv(r'..\datasets\batting_rankings.csv', index=False)
    rank_df_bowling.to_csv(r'..\datasets\bowling_rankings.csv', index=False)
    rank_df_all_rounder.to_csv(r'..\datasets\all_rounder_rankings.csv', index=False)

    return rank_df_batting, rank_df_bowling, rank_df_all_rounder


# %% --------------------------------------------------------------------------
# Function for getting a chosen player's rating and ranking for a chosen month 
# -----------------------------------------------------------------------------

def get_player_rating_ranking(df, player_name, rating_type, year, quarter):

    rank_df = get_rankings(df, rating_type)

    row = rank_df[(rank_df['player_name'] == player_name) & (rank_df['year']==year) & (rank_df['quarter']==quarter)]

    rating = row.at[row.index[0], 'overall_rating_'+rating_type].round(2)
    ranking = int(row.at[row.index[0], 'ranking'])

    result = {'Player Name': player_name, 'Rating': rating, 'Ranking': ranking}

    return result 




