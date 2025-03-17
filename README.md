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


# Key Aspects of the Project Report:
a. Introduction
b. Literature Review (Optional)
c. Dataset(s) Source and Description
d. Data Exploration and Important Features
e. Methods
f. Experimentation
g. Final Results
h. Conclusion
i. References

Essentially the main steps of the Project in terms of the code are as follows:
- Gather the data ( have the source, collect 2010 to 2024 )
- Clean the data, make sure its usable for the model prediction.
- Implement an elo_rating feature, by certain feature engineering methods. An important part of this feature is the subsequent surface_elo_rating.
- Models.ipynb:
    These will include= Linear regression, logistic regression, random forest {essentially}. Obviously there are quite a few aspects to these features involved since we have to build the functions on our own.