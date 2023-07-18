"""
Python module containing functions for engineering features for cricket data

"""

__date__ = "2023-06-17"
__author__ = "SamKemp"


# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

# %% --------------------------------------------------------------------------
# Functions to get batting and bowling dataframes 
# -----------------------------------------------------------------------------

def get_bat_df(df):

    # Create a new dataframe bat_df
    bat_df = df[['event_id', 'innings', 'date', 'batter_id', 'batsman_striker_name', 'is_home_batter', 'batsman_striker_team_name', 'bowler_id', 'bowler_name', 'bowler_team_name', 'batter_balls_faced', 'batter_runs', 'home_score', 'away_score', 'dismissal_dismissal', 'batter_rating', 'bowler_rating']]

    return bat_df


def get_bowl_df(df):

    # Create a new dataframe bowl_df
    bowl_df = df[['event_id', 'innings', 'date', 'bowler_id', 'bowler_name', 'bowler_team_name', 'batter_id', 'batsman_striker_name', 'batsman_striker_team_name', 'is_home_bowler', 'batter_balls_faced', 'batter_runs', 'home_score', 'away_score', 'dismissal_dismissal', 'batter_rating', 'bowler_rating', 'bowler_balls', 'bowler_conceded', 'bowler_wickets', 'innings_no_balls', 'innings_wides']]

    return bowl_df

# %% --------------------------------------------------------------------------
# Function for creating aggregated player data 
# -----------------------------------------------------------------------------

def create_agg_data(df, batter_or_bowler):

    if batter_or_bowler == 'batter':

        # Group the DataFrame 'bat_df' by 'batsman_striker_name' and 'event_id'
        grouped_data = df.groupby(['batsman_striker_name', 'event_id'])

        # Aggregate the grouped data using different aggregation functions for specific columns
        aggregated_batsman_data = grouped_data.agg({
            'batter_runs': 'last',
            'batter_balls_faced': 'last',
            'is_home_batter': 'mean',
            'dismissal_dismissal': 'last',
            'bowler_rating': 'mean'
        }).reset_index()

        return aggregated_batsman_data

    elif batter_or_bowler == 'bowler':

        # Group the DataFrame 'bowl_df' by 'bowler_name' and 'event_id'
        grouped_data = df.groupby(['bowler_name', 'event_id'])

        # Define a custom lambda function to count non-zero entries
        count_non_zero = lambda x: (x != 0).sum()

        # Aggregate the grouped data using different aggregation functions for specific columns
        aggregated_bowler_data = grouped_data.agg({
            'bowler_balls': 'max',
            'bowler_conceded': 'max',
            'bowler_wickets': 'max',
            'innings_wides': count_non_zero,
            'innings_no_balls': count_non_zero,
            'is_home_bowler': 'mean',
            'batter_rating': 'mean'
        }).reset_index()

        return aggregated_bowler_data
    
# %% --------------------------------------------------------------------------
# Function for creating aggregated team data
# -----------------------------------------------------------------------------

def get_agg_team_score_data(df):

    # Group the DataFrame 'df' by 'event_id' and 'innings'
    grouped_data = df.groupby(['event_id', 'bowler_team_id'])

    # Aggregate the grouped data using different aggregation functions for specific columns
    aggregated_innings_score_data = grouped_data.agg({
        'date': 'last',
        'home_score': 'last',
        'away_score': 'last',
        'match_ball_no': 'last'
    }).reset_index()

    # Group the 'aggregated_team_score_data' DataFrame by 'event_id'
    aggregated_team_score_data = aggregated_innings_score_data.groupby('event_id').agg({
        'date': 'last',
        'home_score': 'max',
        'away_score': 'max',
        'match_ball_no': ['last', 'first']
    }).reset_index()

    aggregated_team_score_data = pd.DataFrame(aggregated_team_score_data.values, columns=['event_id', 'date', 'home_score', 'away_score', 'home_balls', 'away_balls'])

    # Create new columns 'home_runs' and 'home_wickets' 
    aggregated_team_score_data['home_runs'] = aggregated_team_score_data['home_score'].str.split('/').str[0]
    aggregated_team_score_data['home_wickets'] = aggregated_team_score_data['away_score'].str.split('/').str[1]

    # Create new columns 'away_runs' and 'away_wickets' 
    aggregated_team_score_data['away_runs'] = aggregated_team_score_data['away_score'].str.split('/').str[0]
    aggregated_team_score_data['away_wickets'] = aggregated_team_score_data['home_score'].str.split('/').str[1]

    return aggregated_team_score_data

# %% --------------------------------------------------------------------------
# Function for merging team and player data
# -----------------------------------------------------------------------------

def merge_with_team_data(aggregated_team_score_data, aggregated_player_data):    
    
    # Merge 'aggregated_team_score_data' and 'aggregated_batsman_data' dataframes on 'event_id'
    df = pd.merge(aggregated_team_score_data, aggregated_player_data, on='event_id')

    return df

# %% --------------------------------------------------------------------------
# Function for creating player won features
# -----------------------------------------------------------------------------

def create_player_won_columns(df, batter_or_bowler):

    # Create a new column named 'home_win'
    df['home_win'] = 0

    # Create a mask based on the condition of home_runs being greater than away_runs
    mask = (df['home_runs'] > df['away_runs'])

    # Assign the match result to the 'home_win' column using the mask
    df.loc[mask, 'home_win'] = 1

    if batter_or_bowler == 'batter':
        # Create a new column named 'batter_won' based on certain conditions
        df['batter_won'] = ((df['is_home_batter'] == 1) & (df['home_win'] == 1)) | ((df['is_home_batter'] == 0) & (df['home_win'] == 0))

    elif batter_or_bowler == 'bowler':
        # Create a new column named 'bowler_won' based on certain conditions
        df['bowler_won'] = ((df['is_home_bowler'] == 1) & (df['home_win'] == 1)) | ((df['is_home_bowler'] == 0) & (df['home_win'] == 0))

    return df
    
# %% --------------------------------------------------------------------------
# Function for games played column
# -----------------------------------------------------------------------------

def create_games_played_column(df, batter_or_bowler):

    if batter_or_bowler == 'batter':
        df['games_played'] = df.groupby('batsman_striker_name').cumcount() + 1

    elif batter_or_bowler == 'bowler':
        df['games_played'] = df.groupby('bowler_name').cumcount() + 1

    return df
        
# %% --------------------------------------------------------------------------
# Function for creating percentage features
# -----------------------------------------------------------------------------

def create_percentage_of_total_feature(df, player_data, home_data, away_data):

    # Convert columns to float type
    df[player_data] = df[player_data].astype(float)
    df[home_data] = df[home_data].astype(float)
    df[away_data] = df[away_data].astype(float) 

    # Define metrics
    metric = player_data.split('_')[1]
    percenatage_metric = '%_of_total_'+ metric

    # Calculate percentage
    df[percenatage_metric] = df[player_data]  / (df[home_data] + df[away_data])

    return df

# %% --------------------------------------------------------------------------
# Function for creating batter run rate feature
# -----------------------------------------------------------------------------

def calculate_batter_run_rate(df):

    # Create a new column named 'run_rate'
    df['run_rate'] = np.where(
        df['batter_balls_faced'] == 0,  # If 'batter_balls_faced' is 0, set the run rate to 0
        0,
        df['batter_runs'] / df['batter_balls_faced']  # Calculate the run rate
    )

    return df

# %% --------------------------------------------------------------------------
# Function for creating bowler run rate feature
# -----------------------------------------------------------------------------

def create_run_rate_against(df):

    # Create a new column 'run_rate_against' 
    df['run_rate_against'] = np.where(
        # If 'bowler_balls' is 0, set the value to NaN
        df['bowler_balls'] == 0,
        np.nan,
        np.where(
            # If 'bowler_conceded' is 0, set the value to 0.5 divided by 'bowler_balls'
            df['bowler_conceded'] == 0,
            0.5 / df['bowler_balls'],
            # Otherwise, calculate the run rate by dividing 'bowler_conceded' by 'bowler_balls'
            df['bowler_conceded'] / df['bowler_balls']
        )
    )

    return df

# %% --------------------------------------------------------------------------
# Function to add metrics to df 
# -----------------------------------------------------------------------------

def add_metrics_to_df(df, batter_or_bowler):

    if batter_or_bowler == 'batter':
        # Create %_of_total_runs column
        df = create_percentage_of_total_feature(df, 'batter_runs', 'home_runs', 'away_runs')
        
        # Create run_rate column
        df = calculate_batter_run_rate(df)

        # Create batter_won column
        df = create_player_won_columns(df, 'batter')

        # Create games_played column
        df = create_games_played_column(df, 'batter')
    
        # Extract necessary columns
        bat_df = df[['event_id', 'date', 'batsman_striker_name', 'batter_runs', 'games_played', '%_of_total_runs', 'run_rate', 'dismissal_dismissal', 'batter_won', 'bowler_rating']]

        return bat_df

    elif batter_or_bowler == 'bowler':
        # Create %_of_total_wickets
        df = create_percentage_of_total_feature(df, 'bowler_wickets', 'home_wickets', 'away_wickets')

        # Create %_of_total_balls
        df = create_percentage_of_total_feature(df, 'bowler_balls', 'home_balls', 'away_balls')

        # Create run_rate_against column
        df = create_run_rate_against(df)

        # Create batter_won column
        df = create_player_won_columns(df, 'bowler')

        # Create games_played column
        df = create_games_played_column(df, 'bowler')

        # Create new dataframe
        bowl_df = df[['event_id', 'date', 'bowler_name', 'bowler_wickets', 'games_played', '%_of_total_wickets', '%_of_total_balls', 'run_rate_against', 'bowler_won', 'batter_rating', 'innings_no_balls', 'innings_wides']]

        return bowl_df
    
# %% --------------------------------------------------------------------------
# Function for creating rating features
# -----------------------------------------------------------------------------

def create_rating_score(df, batter_or_bowler):

    if batter_or_bowler == 'batter':

        # Define our feature rating_score
        df['rating_score'] = df['batter_runs'] * df['%_of_total_runs'] * df['run_rate']

        # Add 30% to the rating_score when batter_won is True
        df.loc[df['batter_won']==True, 'rating_score'] = df.loc[df['batter_won']==True, 'rating_score']* 1.3

        # Add 10% to the rating_score when the batter wasn't dismissed
        df.loc[df['dismissal_dismissal']==False, 'rating_score'] = df.loc[df['dismissal_dismissal']==False, 'rating_score'] * 1.1

        # Calculate the mean of the rating_score column for each player 
        df['player_mean'] = df.groupby('batsman_striker_name')['rating_score'].transform(lambda x: x.expanding().mean())

    if batter_or_bowler == 'bowler':

        # Define our wicket rating score
        df['wicket_rating_score'] = df['bowler_wickets'] * df['%_of_total_wickets'] 

        # Define our run rating score
        df['run_rating_score'] = df['%_of_total_balls'] / df['run_rate_against'] 
        df['run_rating_score'] = df['run_rating_score'].fillna(0)

        df['rating_score'] = (df['wicket_rating_score'] + df['run_rating_score']) / 2

        # Add 30% to the rating_score when batter_won is True
        df.loc[df['bowler_won']==True, 'rating_score'] = df.loc[df['bowler_won']==True, 'rating_score'] * 1.3

        # Take 95% of score for every foul or wide ball
        df['rating_score'] = df['rating_score'] * (0.95**(df['innings_no_balls']+df['innings_wides']))

        # Calculate the mean of the rating_score column for each player 
        df['player_mean'] = df.groupby('bowler_name')['rating_score'].transform(lambda x: x.expanding().mean())

    return df

# %% --------------------------------------------------------------------------
# Function for normalising scores
# -----------------------------------------------------------------------------

def normalise_scores(df, score, lower_bound, upper_bound):

    # Calculate the mean of the score column up to and including the current row
    df['mean'] = df[score].expanding().mean()

    # Calculate the scaling factor to normalize the distribution up to and including the current row
    df['scale_factor'] = 1 / df['mean']

    # Define column name
    score_normalised = score + '_normalised'

    # Apply the scaling factor to the rating_score column up to and including the current row
    df[score_normalised] = df[score] * df['scale_factor']

    # Apply upper and lower bound
    df[score_normalised] = np.clip(df[score_normalised], lower_bound, upper_bound)

    # Drop unnecessary columns
    df = df.drop(['mean', 'scale_factor'], axis=1)

    return df

# %% --------------------------------------------------------------------------
# Function for adding ratings and normalised ratings onto dataframe
# -----------------------------------------------------------------------------

def add_rating_features(df, batter_or_bowler):

    if batter_or_bowler == 'batter':
    
        # Add batter_rating and player_mean
        df = create_rating_score(df, 'batter')
        
        # Normalise rating_score
        df = normalise_scores(df, 'rating_score', 0, 10)

        # Normalise player_mean
        df = normalise_scores(df, 'player_mean', 0.7, 1.3)
        # Set as 1 for the first 10 games played
        df.loc[df['games_played'] <= 10, 'player_mean_normalised'] = 1

        # Normalise bowler_rating
        df = normalise_scores(df, 'bowler_rating', 0.5, 1.5)

    if batter_or_bowler == 'bowler':
        
        # Add bowler_rating and player_mean
        df = create_rating_score(df, 'bowler')

        # Normalise rating_score
        df = normalise_scores(df, 'rating_score', 0, 10)

        # Normalise player_mean
        df = normalise_scores(df, 'player_mean', 0.7, 1.3)
        # Set as 1 for the first 10 games played
        df.loc[df['games_played'] <= 10, 'player_mean_normalised'] = 1

        # Normalise batter_rating
        df = normalise_scores(df, 'batter_rating', 0.5, 1.5)

    return df

# %% --------------------------------------------------------------------------
# Function for combining above functions
# -----------------------------------------------------------------------------

def pre_processing(df, batter_or_bowler):
    
    if batter_or_bowler == 'batter':
    
        bat_df = create_agg_data(df, 'batter')
    
        bat_df = merge_with_team_data(aggregated_team_score_data, bat_df)   

        bat_df = add_metrics_to_df(bat_df, 'batter')
        
        bat_df = create_rating_score(bat_df, 'batter')

        bat_df = add_rating_features(bat_df, 'batter')

        return bat_df

    elif batter_or_bowler == 'bowler':

        bowl_df = create_agg_data(df, 'bowler')

        bowl_df = merge_with_team_data(aggregated_team_score_data, bowl_df)   

        bowl_df = add_metrics_to_df(bowl_df, 'bowler')

        bowl_df = create_rating_score(bowl_df, 'bowler')

        bowl_df = add_rating_features(bowl_df, 'bowler')

        return bowl_df
    
# %% --------------------------------------------------------------------------
# Functions for updating opponent ratings and re-running functions
# -----------------------------------------------------------------------------

def update_rating_score(original_df, bat_df, bowl_df): 

    # Create a bowler_rating score
    bowl_df['bowler_rating_score'] = bowl_df['rating_score_normalised'] * np.log1p(bowl_df['games_played'])

    # Merge witrh original df
    original_df = pd.merge(original_df, bowl_df[['event_id', 'bowler_name', 'bowler_rating_score']], on=['event_id', 'bowler_name'], how='left')
    original_df['bowler_rating'] = original_df['bowler_rating_score']


    # Create a batter_rating score
    bat_df['batter_rating_score'] = bat_df['rating_score_normalised'] * np.log1p(bat_df['games_played'])

    # Merge witrh original df
    original_df = pd.merge(original_df, bat_df[['event_id', 'batsman_striker_name', 'batter_rating_score']], on=['event_id', 'batsman_striker_name'], how='left')
    original_df['batter_rating'] = original_df['batter_rating_score']

    # Get bat_df from the orginal df with the updated bowler_rating 
    bat_df2 = get_bat_df(original_df)

    # Get bowl_df from the orginal df with the updated batter_rating
    bowl_df2 = get_bowl_df(original_df)

    # Run through the pre-processing steps again - batting
    bat_df2 = pre_processing(bat_df2, 'batter')

    # Run through the pre-processing steps again - bowling
    bowl_df2 = pre_processing(bowl_df2, 'bowler')

    # Drop new columns from original df
    original_df.drop(['bowler_rating_score', 'batter_rating_score'], axis=1, inplace=True)

    return bat_df2 , bowl_df2   


# %% --------------------------------------------------------------------------
# Final function for combining all steps to get betting and bowling dataframes
# -----------------------------------------------------------------------------

def create_bat_bowl_dfs(clean_df):
    # Obtain the batting DataFrame
    bat_df = get_bat_df(clean_df)
    
    # Obtain the bowling DataFrame
    bowl_df = get_bowl_df(clean_df)
    
    # Get aggregated team score data (make it global so it can be used in exterior functions)
    global aggregated_team_score_data
    aggregated_team_score_data = get_agg_team_score_data(clean_df)
    
    # Perform pre-processing on the batting DataFrame
    bat_df2 = pre_processing(bat_df, 'batter')
    
    # Perform pre-processing on the bowling DataFrame
    bowl_df2 = pre_processing(bowl_df, 'bowler')
    
    # Update the rating score based on dataframes
    bat_df3, bowl_df3 = update_rating_score(clean_df, bat_df2, bowl_df2)
    
    return bat_df3, bowl_df3
