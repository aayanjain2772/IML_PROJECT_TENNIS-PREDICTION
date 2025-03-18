import pandas as pd
import numpy as np

# Step 1: Load and Prepare Data
df = pd.read_csv("atp_matches_2010_2024_missing_handled.csv")

# Define features (differences between winner and loser)
df["rank_diff"] = df["winner_rank"] - df["loser_rank"]
df["ace_diff"] = df["w_ace"] - df["l_ace"]
df["df_diff"] = df["w_df"] - df["l_df"]
df["svpt_diff"] = df["w_svpt"] - df["l_svpt"]
df["1stIn_diff"] = df["w_1stIn"] - df["l_1stIn"]
df["1stWon_diff"] = df["w_1stWon"] - df["l_1stWon"]
df["2ndWon_diff"] = df["w_2ndWon"] - df["l_2ndWon"]
df["SvGms_diff"] = df["w_SvGms"] - df["l_SvGms"]
df["bpSaved_diff"] = df["w_bpSaved"] - df["l_bpSaved"]
df["bpFaced_diff"] = df["w_bpFaced"] - df["l_bpFaced"]
df["age_diff"] = df["winner_age"] - df["loser_age"]

# Encode categorical variable: surface
df = pd.get_dummies(df, columns=["surface"], drop_first=True)

# Convert bool to int for one-hot encoded columns
for col in ["surface_Grass", "surface_Hard"]:
    df[col] = df[col].astype(int)

# Features to use
feature_cols = [
    "rank_diff", "ace_diff", "df_diff", "svpt_diff", "1stIn_diff", "1stWon_diff",
    "2ndWon_diff", "SvGms_diff", "bpSaved_diff", "bpFaced_diff", "age_diff",
    "surface_Grass", "surface_Hard"  # Assuming Clay is the reference
]

# Debug: Check data types and NaN values in features
print("Data types of features:")
print(df[feature_cols].dtypes)
print("\nNaN check in features:")
print(df[feature_cols].isna().sum())

# Convert features to numeric, coercing errors to NaN
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill NaN values with 0
df[feature_cols] = df[feature_cols].fillna(0)

# Target: 1 if winner is Player A (recorded winner), 0 if loser wins (flip for symmetry later)
df["target"] = 1

# Create a symmetric dataset by flipping winner/loser
df_flipped = df.copy()
df_flipped["rank_diff"] = -df_flipped["rank_diff"]
df_flipped["ace_diff"] = -df_flipped["ace_diff"]
df_flipped["df_diff"] = -df_flipped["df_diff"]
df_flipped["svpt_diff"] = -df_flipped["svpt_diff"]
df_flipped["1stIn_diff"] = -df_flipped["1stIn_diff"]
df_flipped["1stWon_diff"] = -df_flipped["1stWon_diff"]
df_flipped["2ndWon_diff"] = -df_flipped["2ndWon_diff"]
df_flipped["SvGms_diff"] = -df_flipped["SvGms_diff"]
df_flipped["bpSaved_diff"] = -df_flipped["bpSaved_diff"]
df_flipped["bpFaced_diff"] = -df_flipped["bpFaced_diff"]
df_flipped["age_diff"] = -df_flipped["age_diff"]
df_flipped["target"] = 0

# Combine original and flipped data
df_symmetric = pd.concat([df, df_flipped], ignore_index=True)

# Step 2: Train-Test Split
df_symmetric["tourney_date"] = pd.to_datetime(df_symmetric["tourney_date"], format="%Y%m%d", errors="coerce")
df_symmetric = df_symmetric.dropna(subset=["tourney_date"])
df_symmetric["year"] = df_symmetric["tourney_date"].dt.year
train_data = df_symmetric[df_symmetric["year"] <= 2022]
test_data = df_symmetric[df_symmetric["year"] > 2022]

X_train = train_data[feature_cols].values
y_train = train_data["target"].values
X_test = test_data[feature_cols].values
y_test = test_data["target"].values

# Debug: Check array shape and type
print("\nX_train shape:", X_train.shape)
print("X_train dtype:", X_train.dtype)

# Step 3: Normalize features
X_train_mean = np.nanmean(X_train, axis=0)
X_train_std = np.nanstd(X_train, axis=0)
X_train_std[X_train_std == 0] = 1e-10  # Avoid division by zero with small epsilon
X_train = np.where(np.isnan(X_train), 0, X_train)
X_train = (X_train - X_train_mean) / X_train_std
X_test = np.where(np.isnan(X_test), 0, X_test)
X_test = (X_test - X_train_mean) / X_train_std

# Add intercept term
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Step 4: Implement Logistic Regression
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to avoid overflow

def compute_loss(X, y, beta):
    z = np.dot(X, beta)
    y_pred = sigmoid(z)
    loss = -np.mean(y * np.log(y_pred + 1e-10) + (1 - y) * np.log(1 - y_pred + 1e-10))
    return loss

def compute_gradient(X, y, beta):
    z = np.dot(X, beta)
    y_pred = sigmoid(z)
    gradient = np.dot(X.T, (y_pred - y)) / len(y)
    return gradient

# Gradient Descent
beta = np.zeros(X_train.shape[1])  # Initialize coefficients
learning_rate = 0.01
n_iterations = 1000

for i in range(n_iterations):
    gradient = compute_gradient(X_train, y_train, beta)
    beta -= learning_rate * gradient
    if i % 100 == 0:
        loss = compute_loss(X_train, y_train, beta)
        print(f"Iteration {i}, Loss: {loss:.4f}")

# Step 5: Predict and Evaluate
def predict(X, beta):
    z = np.dot(X, beta)
    y_pred = sigmoid(z)
    return (y_pred >= 0.5).astype(int)

# Training accuracy
y_train_pred = predict(X_train, beta)
train_accuracy = np.mean(y_train_pred == y_train)
print(f"\nTraining Accuracy: {train_accuracy:.2%}")

# Test accuracy
y_test_pred = predict(X_test, beta)
test_accuracy = np.mean(y_test_pred == y_test)
print(f"Test Accuracy: {test_accuracy:.2%}")

# Step 6: Save Coefficients and Predictions
coefficients = pd.DataFrame({
    "Feature": ["Intercept"] + feature_cols,
    "Coefficient": beta
})
coefficients.to_csv("logistic_regression_coefficients.csv", index=False)
print("\nLogistic Regression Coefficients:")
print(coefficients)

test_data["predicted"] = y_test_pred
test_data["probability"] = sigmoid(np.dot(X_test, beta))
test_data[["winner_name", "loser_name", "target", "predicted", "probability"]].to_csv("logistic_regression_predictions.csv", index=False)

print("\nLogistic regression complete. Coefficients and predictions saved.")