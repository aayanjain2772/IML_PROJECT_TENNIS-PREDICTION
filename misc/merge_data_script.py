import pandas as pd
import os

# Current directory (where the CSV files are located)
current_dir = os.getcwd()

# List of years to merge (2010–2024)
years = range(2010, 2025)  # 2025 exclusive, so up to 2024

# Initialize an empty list to hold DataFrames
dfs = []

# Loop through each year, read the CSV, and append to the list
for year in years:
    file_path = os.path.join(current_dir, f"atp_matches_{year}.csv")
    if os.path.exists(file_path):
        print(f"Loading {file_path}...")
        df = pd.read_csv(file_path)
        dfs.append(df)
    else:
        print(f"Warning: {file_path} not found. Skipping...")

# Concatenate all DataFrames into one
if dfs:
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Merged {len(dfs)} files into a single DataFrame with {len(merged_df)} rows.")
else:
    raise ValueError("No files were loaded. Check your directory.")

# Optional: Sort by date for chronological order (tourney_date is YYYYMMDD format)
merged_df["tourney_date"] = pd.to_numeric(merged_df["tourney_date"], errors="coerce")
merged_df = merged_df.sort_values("tourney_date", na_position="last")

# Save the merged DataFrame to a new CSV in the current directory
output_file = os.path.join(current_dir, "atp_matches_2010_2024.csv")
merged_df.to_csv(output_file, index=False)
print(f"Saved merged dataset to {output_file} with {len(merged_df)} rows and {len(merged_df.columns)} columns.")

# Optional: Print all columns for reference
print("All columns in merged dataset:", merged_df.columns.tolist())