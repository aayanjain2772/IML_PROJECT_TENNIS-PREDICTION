import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load and process data
df = pd.read_csv("atp_matches_2010_2024_missing_handled.csv")

# First create the difference columns
df["rank_diff"] = df["winner_rank"] - df["loser_rank"]
df["ace_diff"] = df["w_ace"] - df["l_ace"]
df["df_diff"] = df["w_df"] - df["l_df"]
df["1stIn_diff"] = df["w_1stIn"] - df["l_1stIn"]
df["1stWon_diff"] = df["w_1stWon"] - df["l_1stWon"]
df["SvGms_diff"] = df["w_SvGms"] - df["l_SvGms"]
df["bpSaved_diff"] = df["w_bpSaved"] - df["l_bpSaved"]
df["bpFaced_diff"] = df["w_bpFaced"] - df["l_bpFaced"]
df["age_diff"] = df["winner_age"] - df["loser_age"]
df["target"] = 1  # winner is player 1

# Encode surface
df = pd.get_dummies(df, columns=["surface"], drop_first=True)

# Create symmetric dataset
df_flipped = df.copy()
df_flipped["rank_diff"] = -df_flipped["rank_diff"]
df_flipped["ace_diff"] = -df_flipped["ace_diff"]
df_flipped["df_diff"] = -df_flipped["df_diff"]
df_flipped["1stIn_diff"] = -df_flipped["1stIn_diff"]
df_flipped["1stWon_diff"] = -df_flipped["1stWon_diff"]
df_flipped["SvGms_diff"] = -df_flipped["SvGms_diff"]
df_flipped["bpSaved_diff"] = -df_flipped["bpSaved_diff"]
df_flipped["bpFaced_diff"] = -df_flipped["bpFaced_diff"]
df_flipped["age_diff"] = -df_flipped["age_diff"]
df_flipped["target"] = 0  # winner is player 2

# Combine original and flipped data
df_symmetric = pd.concat([df, df_flipped], ignore_index=True)

# Convert tourney_date and split data
df_symmetric["tourney_date"] = pd.to_datetime(df_symmetric["tourney_date"], format="%Y%m%d", errors="coerce")
df_symmetric = df_symmetric.dropna(subset=["tourney_date"])
df_symmetric["year"] = df_symmetric["tourney_date"].dt.year

# Train-test split based on year
train_data = df_symmetric[df_symmetric["year"] <= 2022]
test_data = df_symmetric[df_symmetric["year"] > 2022]

# Features to use
feature_cols = [
    "ace_diff", "df_diff", "1stIn_diff", "SvGms_diff", "age_diff", "1stWon_diff",
    "bpSaved_diff", "bpFaced_diff", "rank_diff", "surface_Grass", "surface_Hard"
]

# Prepare features
X_train = train_data[feature_cols].values
y_train = train_data["target"].values

# Use StandardScaler for normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save the mean and std from the scaler
X_train_mean = scaler.mean_
X_train_std = scaler.scale_

# Add intercept term
X_train_scaled = np.hstack([np.ones((X_train_scaled.shape[0], 1)), X_train_scaled])

# Train logistic regression
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def compute_gradient(X, y, beta):
    z = np.dot(X, beta)
    y_pred = sigmoid(z)
    gradient = np.dot(X.T, (y_pred - y)) / len(y)
    return gradient

# Gradient Descent
beta = np.zeros(X_train_scaled.shape[1])
learning_rate = 0.01
n_iterations = 5000

print("Training model...")
for i in range(n_iterations):
    gradient = compute_gradient(X_train_scaled, y_train, beta)
    beta -= learning_rate * gradient
    if i % 1000 == 0:
        print(f"Iteration {i}")

# Save model parameters
print("Saving model parameters...")
np.save('model_coefficients.npy', beta)
np.save('X_train_mean.npy', X_train_mean)
np.save('X_train_std.npy', X_train_std)

print("Model parameters saved successfully!") 