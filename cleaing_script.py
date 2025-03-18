# methods of handling missing values include: 
# Imputation, Drop Rows, Drop Columns

import pandas as pd
# for dataframe related manipulations
import matplotlib as plt
# for displaying certain features and their aspects

# load the merged dataset into a df and call it 'merged_df'
merged_df = pd.read_csv("atp_matches_2010_2024.csv")
print("Initial data types:")
print(merged_df.dtypes)

# first check which numerical features have missing values:
# first figure out numerical columns = 
numerical_cols = [] # define
categorical_cols = [] # define
# iterate thru the columns in the dataframe to figure out which are float/int
for col in merged_df.columns:
    dtype = merged_df[col].dtype
    unique_count = merged_df[col].nunique()
    total_count = len(merged_df)

    if dtype in ["int64", "float64"]:
        if col in ["winner_id", "loser_id", "tourney_id", "match_num"]:
            categorical_cols.append(col)
        elif unique_count < 10: 
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    else:
        categorical_cols.append(col)
# semantic classification
if "tourney_date" in categorical_cols:
    categorical_cols.remove("tourney_date")
    numerical_cols.append("tourney_date")

# print the classifications:
print("\nNumerical Columns:")
print(numerical_cols)
print(f"Total numerical columns: {len(numerical_cols)}\n")

print("Categorical Columns:")
print(categorical_cols)
print(f"Total categorical columns: {len(categorical_cols)}\n")

# Calculate missing values for numerical columns
numerical_missing = merged_df[numerical_cols].isnull().sum()
numerical_missing_percent = (merged_df[numerical_cols].isnull().mean() * 100).round(2)

# Combine into a DataFrame for reporting
missing_numerical_df = pd.DataFrame({
    "Missing Count": numerical_missing,
    "Missing Percent": numerical_missing_percent
})
missing_numerical_df = missing_numerical_df[missing_numerical_df["Missing Count"] > 0]
print("Numerical columns with missing values:")
print(missing_numerical_df)

# fill missing values of certian features with a default variable.
merged_df["w_ace"]