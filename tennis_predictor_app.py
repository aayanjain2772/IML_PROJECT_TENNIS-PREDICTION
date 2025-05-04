import streamlit as st
import pandas as pd
import numpy as np
import os

# Load the player stats data
@st.cache_data
def load_player_stats():
    # Load your Excel file with player stats from the Desktop
    file_path = "/Users/Agriya/Desktop/betting_tennis_journal.xlsx"
    try:
        # Read the raw Excel file
        df = pd.read_excel(file_path, sheet_name='calc sheet')
        
        # Create a list to store player data
        player_data = []
        
        # Process each row
        for index, row in df.iterrows():
            # Skip empty rows
            if pd.isna(row[0]):
                continue
                
            # Create player dictionary with stats
            player_dict = {
                'Player Name:': row[0],  # First column contains player name
                'Matches Played': row[1],
                'Aces': row[2],
                'DFs': row[3],
                '1stIn': row[4],
                'SvGms': row[5],
                'age': 25,  # Default age if not available
                '1stWon': row[6],
                'bpSaved': row[7],
                'bpFaced': row[8],
                'rank': index + 1,  # Using row number as rank
                'Aces %': row[9],
                'DF %': row[10]
            }
            player_data.append(player_dict)
        
        # Convert to DataFrame
        player_stats_df = pd.DataFrame(player_data)
        
        # Display the processed data for verification
        st.write("Processed Player Stats:")
        st.dataframe(player_stats_df.head())
        
        return player_stats_df
        
    except FileNotFoundError:
        st.error(f"Could not find Excel file at: {file_path}")
        st.stop()
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        st.stop()

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def predict_proba(X, beta):
    z = np.dot(X, beta)
    y_pred = sigmoid(z)
    return np.column_stack([1 - y_pred, y_pred])

# Load model parameters
try:
    beta = np.load('model_coefficients.npy')
    X_train_mean = np.load('X_train_mean.npy')
    X_train_std = np.load('X_train_std.npy')
except Exception as e:
    st.error(f"Error loading model parameters: {e}")
    st.stop()

# Streamlit UI
st.title('Tennis Match Outcome Predictor')

# Load player stats
try:
    player_stats_df = load_player_stats()
    player_names = sorted(player_stats_df['Player Name:'].tolist())
except Exception as e:
    st.error(f"Error loading player stats: {e}")
    st.stop()

# Create two columns for player selection
col1, col2 = st.columns(2)

with col1:
    st.subheader("Player 1")
    player1_name = st.selectbox("Select Player 1", [""] + player_names, key='p1')

with col2:
    st.subheader("Player 2")
    player2_name = st.selectbox("Select Player 2", [""] + player_names, key='p2')

# Surface selection
surface = st.selectbox("Select Surface", ["Clay", "Grass", "Hard"])

# Predict button
if st.button("Predict Match Outcome"):
    if not player1_name or not player2_name:
        st.warning("Please select both players")
    elif player1_name == player2_name:
        st.warning("Please select different players")
    else:
        try:
            # Get player stats
            p1_stats = player_stats_df[player_stats_df['Player Name:'] == player1_name].iloc[0]
            p2_stats = player_stats_df[player_stats_df['Player Name:'] == player2_name].iloc[0]

            # Calculate differences
            ace_diff = float(p1_stats['Aces']) - float(p2_stats['Aces'])
            df_diff = float(p1_stats['DFs']) - float(p2_stats['DFs'])
            firstIn_diff = float(p1_stats['1stIn']) - float(p2_stats['1stIn'])
            SvGms_diff = float(p1_stats['SvGms']) - float(p2_stats['SvGms'])
            age_diff = float(p1_stats['age']) - float(p2_stats['age'])
            firstWon_diff = float(p1_stats['1stWon']) - float(p2_stats['1stWon'])
            bpSaved_diff = float(p1_stats['bpSaved']) - float(p2_stats['bpSaved'])
            bpFaced_diff = float(p1_stats['bpFaced']) - float(p2_stats['bpFaced'])
            rank_diff = float(p1_stats['rank']) - float(p2_stats['rank'])

            # Surface encoding
            surface_Grass = 1 if surface == "Grass" else 0
            surface_Hard = 1 if surface == "Hard" else 0

            # Create feature vector
            features = np.array([[
                ace_diff, df_diff, firstIn_diff, SvGms_diff, age_diff,
                firstWon_diff, bpSaved_diff, bpFaced_diff, rank_diff,
                surface_Grass, surface_Hard
            ]])

            # Normalize features
            features_norm = (features - X_train_mean) / X_train_std
            features_norm = np.hstack([np.ones((features_norm.shape[0], 1)), features_norm])

            # Make prediction
            probs = predict_proba(features_norm, beta)
            
            # Display results
            st.subheader("Prediction Results")
            
            # Create columns for displaying probabilities
            prob_col1, prob_col2 = st.columns(2)
            
            with prob_col1:
                st.metric(
                    label=f"{player1_name} Win Probability",
                    value=f"{probs[0,1]*100:.1f}%"
                )
            
            with prob_col2:
                st.metric(
                    label=f"{player2_name} Win Probability",
                    value=f"{probs[0,0]*100:.1f}%"
                )

            winner = player1_name if probs[0,1] > 0.5 else player2_name
            st.success(f"Predicted Winner: {winner}")

            # Display statistics comparison
            st.subheader("Player Statistics Comparison")
            
            stats_df = pd.DataFrame({
                'Statistic': ['Aces', 'Double Faults', '1st Serve In', 'Service Games', 
                             'Age', '1st Serve Won', 'Break Points Saved', 
                             'Break Points Faced', 'Rank', 'Aces %', 'Double Faults %'],
                player1_name: [
                    p1_stats['Aces'], p1_stats['DFs'], p1_stats['1stIn'],
                    p1_stats['SvGms'], p1_stats['age'], p1_stats['1stWon'],
                    p1_stats['bpSaved'], p1_stats['bpFaced'], p1_stats['rank'],
                    p1_stats['Aces %'], p1_stats['DF %']
                ],
                player2_name: [
                    p2_stats['Aces'], p2_stats['DFs'], p2_stats['1stIn'],
                    p2_stats['SvGms'], p2_stats['age'], p2_stats['1stWon'],
                    p2_stats['bpSaved'], p2_stats['bpFaced'], p2_stats['rank'],
                    p2_stats['Aces %'], p2_stats['DF %']
                ],
                'Difference': [
                    ace_diff, df_diff, firstIn_diff, SvGms_diff, age_diff,
                    firstWon_diff, bpSaved_diff, bpFaced_diff, rank_diff,
                    float(p1_stats['Aces %']) - float(p2_stats['Aces %']),
                    float(p1_stats['DF %']) - float(p2_stats['DF %'])
                ]
            })
            
            st.dataframe(stats_df.style.highlight_positive(subset=['Difference'], color='lightgreen')
                                    .highlight_negative(subset=['Difference'], color='lightpink'))

        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.error("Please check if all required statistics are available for both players") 