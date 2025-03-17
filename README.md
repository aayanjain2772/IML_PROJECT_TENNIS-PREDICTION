# IML_PROJECT_TENNIS-PREDICTION

1. Dataset Source and Description:
Our source is this link: https://github.com/JeffSackmann/tennis_atp/tree/master
We will be using data from 2010 to 2024
Our goal is to predict Wimbledone 2025
## Features: 
tourney_date, surface, winner_name, loser_name, winner_rank, loser_rank, w_ace, l_ace, etc.
## Raw State: 
Unprocessed with missing stats (e.g., aces), inconsistent names, and no precomputed ratings.

# One Key FEature Engineering aspect:
Making the elo ratings for the players. this will; serve as a key parameter as we will see in the data exploration.

# Models:
Linear regression
Logistic regression
Random forest