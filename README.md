<h1 align="center">Predictive Modeling of ATP Tennis Outcomes Using Custom Elo Ratings: A Manual Implementation</h1>

###

<h2 align="left">a) Introduction</h2>

###

<p align="left">Blah blah blah, we will predict tennis matches using AI and ML learned in class. rough rough rough</p>

###

<h2 align="left">b) Dataset(s) Sources and Description:</h2>

###

<p align="left">The primary dataset for this project is sourced from Jeff Sackmann’s 'tennis_atp' GitHub repository (Sackmann, n.d.), a widely recognized open-access collection of raw historical match data for the Association of Tennis Professionals (ATP) men’s professional tennis circuit. This repository, available at https://github.com/JeffSackmann/tennis_atp/tree/master, provides annual CSV files (e.g., atp_matches_2010.csv through atp_matches_2024.csv) containing detailed records of ATP matches spanning from the 1960s to the present, with a focus in this project on the recent period of 2018–2023 to ensure relevance and manage computational scope. Each file encapsulates match-level data, including tournament details, player identities, outcomes, and performance statistics, making it an ideal raw resource for predictive modeling as inspired by Green Code’s (2025) approach to tennis match prediction.

An important aspect of this project in terms of the Dataset will also include cleaning our dataset. Some of the things we will work in cleaning our dataset is: Handling missing values, Standardizing inconsistent data, Removing duplicates, Correcting data types, Handling outliers and Invalid data, Normalizing and transforming data, and Profiling and logging.

So we used a jupyter notebook to handle cleaning, and I think the correct way to explore the data would be through a jupyter notebook itself, we'll switch to .py files when we reach models.</p>

###

<h2 align="left">c) Data Exploration and Important Feeatures:</h2>

###

<p align="left">In the previous step, we just encorporated handling missing values and cleaned our data. Now the handled .csv file contains no missing values and we can do our data exploration now. The goal of data exploration is to: understand the data, its distribution, datatypes, shape, identify outliers, skewness, patterns and relationships, examine the range, central tendency, look for correlations, detect anamolies, find errors, inform feature engineering, and ultimately generate insights.

A key aspect in our project is the aspect of coming up with a Custom Elo rating algorithm, here we will call it the TennisElo. For someone who isn't familiar with the Elo rating, they can refer to this page: https://www.chess.com/terms/elo-rating-chess; but essentially it's an algorithm used to rank players in Chess. We have implemented its functionality for our project, but have modified its logic specific to tennis. We have also defined its parameters seperate from chess, since it's a different game. An Elo rating is essentially a comparative predictive model. Now for the sake of this project we will not go very in-depth into the aspects of sets played, game formats and other aspects because we have kept the logic same throughout our overall work. If someone were to internsively research on TennisElo and creating predictive models on the described dataset, we would suggest them to look into depths such as specific surface ratings and other aspects. One challenge we encountered was handling the entrance of new players into our dataset, which we will explain here. Overall, in our opinion and through extensive research, the Elo Rating is valid approach to figuring out tennis matches and playerVplayer matchups.
</p>

<h5 align="left">the Tennis Elo</h5>

<p align="left">
- Elo rating formula:
</p>
' $$ E_A = 1/(1 + 10^(R_B - R_A)/400) $$'
<p align="left">
- Base rating:
- Rating updating rule:
- Before and after match ratings
- Elo parameters
</p>

###

<h2 align="left">d) Methods:</h2>

###

<p align="left">The primary dataset for this project is sourced from Jeff Sackmann’s 'tennis_atp' GitHub repository (Sackmann, n.d.), a widely recognized open-access collection of raw historical match data for the Association of Tennis Professionals (ATP) men’s professional tennis circuit. This repository, available at https://github.com/JeffSackmann/tennis_atp/tree/master, provides annual CSV files (e.g., atp_matches_2010.csv through atp_matches_2024.csv) containing detailed records of ATP matches spanning from the 1960s to the present, with a focus in this project on the recent period of 2018–2023 to ensure relevance and manage computational scope. Each file encapsulates match-level data, including tournament details, player identities, outcomes, and performance statistics, making it an ideal raw resource for predictive modeling as inspired by Green Code’s (2025) approach to tennis match prediction.</p>

###

<h2 align="left">e) Experimentation:</h2>

###

<p align="left">The primary dataset for this project is sourced from Jeff Sackmann’s 'tennis_atp' GitHub repository (Sackmann, n.d.), a widely recognized open-access collection of raw historical match data for the Association of Tennis Professionals (ATP) men’s professional tennis circuit. This repository, available at https://github.com/JeffSackmann/tennis_atp/tree/master, provides annual CSV files (e.g., atp_matches_2010.csv through atp_matches_2024.csv) containing detailed records of ATP matches spanning from the 1960s to the present, with a focus in this project on the recent period of 2018–2023 to ensure relevance and manage computational scope. Each file encapsulates match-level data, including tournament details, player identities, outcomes, and performance statistics, making it an ideal raw resource for predictive modeling as inspired by Green Code’s (2025) approach to tennis match prediction.</p>

###

<h2 align="left">f) Final Results:</h2>

###

<p align="left">The primary dataset for this project is sourced from Jeff Sackmann’s 'tennis_atp' GitHub repository (Sackmann, n.d.), a widely recognized open-access collection of raw historical match data for the Association of Tennis Professionals (ATP) men’s professional tennis circuit. This repository, available at https://github.com/JeffSackmann/tennis_atp/tree/master, provides annual CSV files (e.g., atp_matches_2010.csv through atp_matches_2024.csv) containing detailed records of ATP matches spanning from the 1960s to the present, with a focus in this project on the recent period of 2018–2023 to ensure relevance and manage computational scope. Each file encapsulates match-level data, including tournament details, player identities, outcomes, and performance statistics, making it an ideal raw resource for predictive modeling as inspired by Green Code’s (2025) approach to tennis match prediction.</p>

###

<h2 align="left">g) Conclusions:</h2>

###

<p align="left">The primary dataset for this project is sourced from Jeff Sackmann’s 'tennis_atp' GitHub repository (Sackmann, n.d.), a widely recognized open-access collection of raw historical match data for the Association of Tennis Professionals (ATP) men’s professional tennis circuit. This repository, available at https://github.com/JeffSackmann/tennis_atp/tree/master, provides annual CSV files (e.g., atp_matches_2010.csv through atp_matches_2024.csv) containing detailed records of ATP matches spanning from the 1960s to the present, with a focus in this project on the recent period of 2018–2023 to ensure relevance and manage computational scope. Each file encapsulates match-level data, including tournament details, player identities, outcomes, and performance statistics, making it an ideal raw resource for predictive modeling as inspired by Green Code’s (2025) approach to tennis match prediction.</p>

###

<h2 align="left">h) References:</h2>

###

<p align="left">
- https://fivethirtyeight.com/features/how-should-we-rate-a-tennis-player/
- https://www.sciencedirect.com/science/article/abs/pii/S0377221721003234?via%3Dihub
- https://www.tennisabstract.com/blog/2019/12/03/an-introduction-to-tennis-elo/ 
- https://tennisabstract.com/reports/atp_elo_ratings.html 
- https://gwern.net/doc/statistics/order/comparison/1978-elo-theratingofchessplayerspastandpresent.pdf 
</p>

###

<div align="left">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="40" alt="python logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="40" alt="github logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original.svg" height="40" alt="jupyter logo"  />
</div>

###