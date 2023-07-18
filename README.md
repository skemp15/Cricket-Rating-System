# Ball-by-Ball T20 Cricket Data Analysis and Player Rating System

## Introduction

This project aims to analyse ball-by-ball T20 cricket data and develop a player rating system. The primary goal is to evaluate players' performances based on their batting and bowling statistics and create a comprehensive rating system. The project involved cleaning a vast dataset with over a million entries and 89 features, ensuring data consistency, and identifying relevant metrics to rate players' performance.

## Data Cleaning

The initial phase of the project involved thorough data cleaning. The dataset contained vast amounts of data with numerous features, including player names, match details, and performance metrics. Some of the steps taken during the data cleaning process included:

- **Filtering T20 Cricket:** Removed any entries that did not correspond to T20 cricket matches.
- **Handling Missing Values:** Filled in null values in a logical manner to avoid distortions in the analysis.
- **Data Consistency:** Ensured that important features, such as player names, were consistent throughout the dataset.

## Identifying Metrics

To create an effective player rating system, the metrics for batting and bowling performances were identified. The batting metrics included total runs and run rate, while bowling metrics included wickets taken and run rate against.

## Scoring System

Based on the identified metrics, a scoring system was developed to evaluate each player's performance in each game. This scoring system provided individual scores for batting and bowling performances. The next step involved using these scores for each match to create an overall rating for the player based on all of their past performances. I did this through using a weighted average algorithm to calculate each player's overall rating.

The weighted average algorithm considered factors such as the number of games played to penalise players with fewer appearances, while also rewarding or penalising players for periods of good or bad form, respectively.

## Aggregation and Visualisation

After calculating the ratings for each player in every game, the data was aggregated to display monthly ratings for each player. A Power BI dashboard was created to visualise the results effectively. This dashboard allowed users to examine the top-rated players at any given point in time and track how a particular player's rating changed over time.

## Insights and Limitations

Working on this project was an exciting experience as it required developing a custom algorithm rather than relying on pre-made machine learning models. However, certain limitations were identified during the analysis:

1. **Long-standing Players:** The algorithm tended to give too much weight to players with extensive careers, possibly overlooking rising talents.
2. **Difficulty in Rating Drops:** Players who achieved high ratings found it challenging to lose their positions, particularly due to the absence of penalties for extended periods of inactivity or retired players.
3. **Anomalies in Bowling Ratings:** The bowling rating system displayed certain anomalies, ranking players with very few matches played among the top-rated performers.

## Conclusion

Despite the identified limitations, I am proud of my attempt to design and implement the player rating system. The project allowed me to develop my knowledge of Python and data science in a new and interesting way by devising a custom algorithm for data analysis. I received valuable feedback and believe that with more time and a better understanding of cricket, the system could be further improved. However, I believe this project showcased my ability to handle large and complex datasets, use my initiative  when thinking about what metrics are important for a certain problem, and allowed me to create interesting visualisations using Power BI that allowed me to identify the problems with my approach that could hypothetically be improved on in the future. 
