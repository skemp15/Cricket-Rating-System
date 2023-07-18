"""
Python module containing functions for pre-processing and cleaning cricket data

"""

__date__ = "2023-06-16"
__author__ = "SamKemp"


# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import datetime

# %% --------------------------------------------------------------------------
# Function for creating subset of dataframe
# -----------------------------------------------------------------------------

def filter_dataframe_by_date(df, lower_date=None, upper_date=None):
    
    if df['date'].dtype == 'object':
        # Convert dates to datetime
        df['date'] = df['date'].str[:10]
        df['date'] = pd.to_datetime(df['date'], format='mixed')

    # Convert lower_date and upper_date to pandas datetime if they are strings
    if isinstance(lower_date, str):
        lower_date = pd.to_datetime(lower_date)
    if isinstance(upper_date, str):
        upper_date = pd.to_datetime(upper_date)

    # Determine the earliest and latest dates in the dataframe
    min_date = df['date'].min()
    max_date = df['date'].max()

    # Assign the earliest or latest date if lower_date or upper_date is not provided
    lower_date = lower_date or min_date
    upper_date = upper_date or max_date

    # Filter the dataframe based on the date range
    filtered_dataframe = df[(df['date'] >= lower_date) & (df['date'] <= upper_date)]

    return filtered_dataframe

# %% --------------------------------------------------------------------------
# Function to check number of names for each id
# -----------------------------------------------------------------------------

def check_duplicate_names_for_id(df, id, name):
    num_duplicate_names = (df.groupby(id)[name].nunique() > 1).sum()
    return num_duplicate_names

# %% --------------------------------------------------------------------------
# Function to replace duplicate names in the DataFrame for a given id and name column
# -----------------------------------------------------------------------------

def replace_duplicate_names(df, id, name):
    
    num_duplicates = check_duplicate_names_for_id(df, id, name)
    
    if num_duplicates == 0:
        print('No duplicate name(s) for:', id)
    else:        
        print(f'{num_duplicates} duplicate names in {id}')
        
        # Get the ids with the most duplicate names, sorted in descending order
        duplicate_list = df.groupby(id)[name].nunique().sort_values(ascending=False).head(num_duplicates)
        
        # Iterate through the duplicate ids
        for duplicate_id in duplicate_list.index:
            # Get the unique names for the current duplicate id, sorted by frequency
            unique_names = df[df[id] == duplicate_id][name].value_counts().sort_values(ascending=False).index
            
            # Select the most common name as the top name
            top_name = unique_names[0]
            
            # Iterate through the remaining unique names
            for player_name in unique_names[1:]:
                # Replace each unique name with the top name in the dataframe
                df[name] = df[name].replace({player_name: top_name})
                
                print(f'{player_name} replaced with {top_name}')
    
    return df

# %% --------------------------------------------------------------------------
# Function to fill names from name ID
# -----------------------------------------------------------------------------

def fill_name_from_id(df, id, name):

    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        if pd.isnull(row[name]):
            # Retrieve the bowler_id for the current row
            player_id = row[id]

            # Find matching names for the bowler_id
            matching_names = df.loc[df[id] == player_id, name]

            # If matching names exist, assign the first matching name to the current row's 'bowler_name'
            if not matching_names.empty:
                df.at[index, name] = matching_names.values[0]

    return df

# %% --------------------------------------------------------------------------
# Function to replace discrepancies in specified columns with 'Unknown' values in the DataFrame.
# -----------------------------------------------------------------------------

def replace_discrepancies_with_unknown(df, discrepancy_column, column_to_replace1, column_to_replace2):

    # Group the DataFrame by 'event_id' and the first replacement column, and count the number of unique values in the discrepancy column
    discrepancy_df = df.groupby(['event_id', column_to_replace1])[discrepancy_column].nunique().loc[lambda x: x > 1].reset_index()[['event_id', column_to_replace1]]

    for i, row in discrepancy_df.iterrows():
        
        # Define event_id and batter_id
        event_id = row['event_id']
        id = row[column_to_replace1]

        if id == 'Unknown':
            continue

        else:

            # Filter the DataFrame based on the specified conditions
            filtered_df = df.loc[(df['event_id'] == event_id) & (df[column_to_replace1] == id)]

            # Calculate the most common value
            most_common_value = filtered_df[discrepancy_column].mode().values[0]

            # Replace any value except the most common value with 'Unknown Batter'
            filtered_df.loc[filtered_df[discrepancy_column] != most_common_value, column_to_replace1] = 'Unknown'
            filtered_df.loc[filtered_df[discrepancy_column] != most_common_value, column_to_replace2] = 'Unknown'

            # Update the original DataFrame with the modified values
            df.update(filtered_df)
    
    return df


# %% --------------------------------------------------------------------------
# Combine all pre-processing
# -----------------------------------------------------------------------------

def combine_pre_processing(df):

    if df['date'].dtype == 'object':
        # Convert dates to datetime
        df['date'] = df['date'].str[:10]
        df['date'] = pd.to_datetime(df['date'], format='mixed')

    # Identify event IDs with overs greater than 20
    event_ids_to_remove = df.loc[df['overs'] > 20, 'event_id'].unique()

    # Remove rows with event IDs that have overs greater than 20
    df = df.loc[~df['event_id'].isin(event_ids_to_remove)]

    # Identify event IDs with overs greater than 20
    event_ids_to_remove = df.loc[df['over_limit'] > 20, 'event_id'].unique()

    # Remove rows with event IDs that have overs greater than 20
    df = df.loc[~df['event_id'].isin(event_ids_to_remove)]

    # Create a dictionary of player id and name pairs
    role_id_names = {'bowler_id': 'bowler_name',
                'batter_id': 'batsman_striker_name',
                'nonstriker_id': 'batsman_nonstriker_name',
                'dismissal_bowler_id': 'dismissal_bowler_name',
                'dismissal_batsman_id': 'dismissal_batsman_name'}

    # Iterate through the dictionary and replace duplicates
    for id, name in role_id_names.items():
        replace_duplicate_names(df, id, name)
        print('\n')

    # Create a dictionary of team id and name pairs
    team_id_names = {'bowler_team_id': 'bowler_team_name',
                'batsman_striker_team_id': 'batsman_striker_team_name',
                'batsman_nonstriker_team_id': 'batsman_nonstriker_team_name'}

    # Iterate through the dictionary and replace duplicates
    for id, name in team_id_names.items():
        replace_duplicate_names(df, id, name)
        print('\n')

    # Drop rows where 'id' matches 999999999999999
    df = df.drop(df[df['id'] == 999999999999999].index)

    # Forward fill missing values based on 'event_id', 'innings', and the same value before the decimal point for 'overs'
    df['bowler_id'] = df.groupby(['event_id', 'innings', df['overs'].apply(int)])['bowler_id'].fillna(method='ffill')

    # Use the function to fill missing names from ID
    fill_name_from_id(df, 'bowler_id', 'bowler_name')

    # Fill missing values in 'bowler_id' and 'bowler_name' columns with 'Unknown'
    df['bowler_id'] = df['bowler_id'].fillna('Unknown')
    df['bowler_name'] = df['bowler_name'].fillna('Unknown')

    # Create a mask based on specific conditions
    mask = (
        ((df['score_value'].shift(1) % 2 == 0) & (df['wickets_lost'].shift(1) == 0)) &  # If the previous score is even and no wickets were lost
        (df['event_id'].shift(1) == df['event_id']) &  # If the previous event_id matches the current event_id
        (df['innings'].shift(1) == df['innings'])  # If the previous innings value matches the current innings value
    )

    # Initialise variables for missing count
    prev_missing_count = float('inf')
    curr_missing_count = df['batter_id'].isna().sum()

    # Iteratively fill missing values until the count stabilizes
    while curr_missing_count != prev_missing_count:
        prev_missing_count = curr_missing_count

        # Fill missing values in the 'batter_id' column using the previous non-missing value
        df.loc[mask, 'batter_id'] = df['batter_id'].fillna(df['batter_id'].shift(1))

        # Update the current missing count
        curr_missing_count = df['batter_id'].isna().sum()

    # Use the function to fill missing names from ID
    fill_name_from_id(df, 'batter_id', 'batsman_striker_name')

    # Fill missing values in 'batter_id' and 'batsman_striker_name' columnn with 'Unknown'
    df['batter_id'] = df['batter_id'].fillna('Unknown')
    df['batsman_striker_name'] = df['batsman_striker_name'].fillna('Unknown')

    # Set the initial value of the 'is_home_batter' column to 0 for all rows
    df['is_home_batter'] = 0

    # Update the 'is_home_batter' column to 1 for rows where the 'away_score' is '0'
    df.loc[df['away_score'] == '0', 'is_home_batter'] = 1

    df = replace_discrepancies_with_unknown(df, 'is_home_batter', 'batter_id', 'batsman_striker_name')

    # Assign 0 to the 'is_home_bowler' column for all rows in the DataFrame
    df['is_home_bowler'] = 0

    # Set the value of 'is_home_bowler' to 1 for rows where the 'home_score' column is '0'
    df.loc[df['home_score'] == '0', 'is_home_bowler'] = 1

    # Run the above function on discrepancies with is_home_bowler
    df = replace_discrepancies_with_unknown(df, 'is_home_bowler', 'bowler_id', 'bowler_name')

    # Step 2: Filter event_ids with only one unique innings value
    unique_innings = df.groupby('event_id')['innings'].nunique()
    filtered_event_ids = unique_innings[unique_innings == 1].index

    # Step 3: Create a new DataFrame without the filtered event_ids
    df = df[~df['event_id'].isin(filtered_event_ids)]

    # We can set preliminary batter and bowling ratings for future use 
    df['bowler_rating'] = 1
    df['batter_rating'] = 1

    # Finally we can reset the index
    df = df.reset_index(drop=True)

    return df
