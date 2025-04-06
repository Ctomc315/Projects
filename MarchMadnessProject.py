import glob
import pandas as pd
import numpy as np
import os
import pandas as pd
import sklearn

# 1. Load men's detailed regular season results and (optionally) tournament results
m_reg = pd.read_csv('~/Desktop/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv')
tourney_results = pd.read_csv('~/Desktop/march-machine-learning-mania-2025/MNCAATourneyDetailedResults.csv')  # Optional

# 2. Extract and rename winner stats into a unified format
winners_df = m_reg[[
    'Season', 'WTeamID', 'WScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3',
    'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF'
]].copy()
winners_df.rename(columns={
    'WTeamID': 'TeamID',
    'WScore':  'Score',
    'WFGM':    'FGM',
    'WFGA':    'FGA',
    'WFGM3':   'FGM3',
    'WFGA3':   'FGA3',
    'WFTM':    'FTM',
    'WFTA':    'FTA',
    'WOR':     'OR',
    'WDR':     'DR',
    'WAst':    'Ast',
    'WTO':     'TO',
    'WStl':    'Stl',
    'WBlk':    'Blk',
    'WPF':     'PF'
}, inplace=True)

# 3. Extract and rename loser stats into a unified format
losers_df = m_reg[[
    'Season', 'LTeamID', 'LScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3',
    'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF'
]].copy()
losers_df.rename(columns={
    'LTeamID': 'TeamID',
    'LScore':  'Score',
    'LFGM':    'FGM',
    'LFGA':    'FGA',
    'LFGM3':   'FGM3',
    'LFGA3':   'FGA3',
    'LFTM':    'FTM',
    'LFTA':    'FTA',
    'LOR':     'OR',
    'LDR':     'DR',
    'LAst':    'Ast',
    'LTO':     'TO',
    'LStl':    'Stl',
    'LBlk':    'Blk',
    'LPF':     'PF'
}, inplace=True)

# 4. Concatenate winner and loser DataFrames into one
team_stats_df = pd.concat([winners_df, losers_df], axis=0)

# 5. Aggregate stats per (Season, TeamID) using mean (change to sum if needed)
team_agg = team_stats_df.groupby(['Season', 'TeamID'], as_index=False).agg({
    'Score': 'mean',
    'FGM':   'mean',
    'FGA':   'mean',
    'FGM3':  'mean',
    'FGA3':  'mean',
    'FTM':   'mean',
    'FTA':   'mean',
    'OR':    'mean',
    'DR':    'mean',
    'Ast':   'mean',
    'TO':    'mean',
    'Stl':   'mean',
    'Blk':   'mean',
    'PF':    'mean'
})

# Rename aggregated columns for clarity
team_agg.rename(columns={
    'Score': 'AvgScore',
    'FGM': 'AvgFGM',
    'FGA': 'AvgFGA',
    'FGM3': 'AvgFGM3',
    'FGA3': 'AvgFGA3',
    'FTM': 'AvgFTM',
    'FTA': 'AvgFTA',
    'OR': 'AvgOR',
    'DR': 'AvgDR',
    'Ast': 'AvgAst',
    'TO': 'AvgTO',
    'Stl': 'AvgStl',
    'Blk': 'AvgBlk',
    'PF': 'AvgPF'
}, inplace=True)

# 6. Calculate win-loss records for men's games
# Count wins
wins_df = m_reg.groupby(['Season', 'WTeamID']).size().reset_index(name='Wins')
wins_df.rename(columns={'WTeamID': 'TeamID'}, inplace=True)

# Count losses
losses_df = m_reg.groupby(['Season', 'LTeamID']).size().reset_index(name='Losses')
losses_df.rename(columns={'LTeamID': 'TeamID'}, inplace=True)

# Merge wins and losses and compute win percentage
wl_df = pd.merge(wins_df, losses_df, on=['Season', 'TeamID'], how='outer').fillna(0)
wl_df['Games'] = wl_df['Wins'] + wl_df['Losses']
wl_df['WinPct'] = wl_df['Wins'] / wl_df['Games']

# 7. Merge aggregated stats with win-loss records to form final team features
team_features = pd.merge(team_agg, wl_df, on=['Season', 'TeamID'], how='left')

# Display the first few rows of the final DataFrame
print(team_features.head())

# From the original regular season data (m_reg), extract opponent stats

# When a team wins, opponent stats come from the loser columns.
winners_opponents = m_reg[[
    'Season', 'WTeamID', 'LScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3',
    'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF'
]].copy()
winners_opponents.rename(columns={
    'WTeamID': 'TeamID',
    'LScore': 'OppScore',
    'LFGM': 'OppFGM',
    'LFGA': 'OppFGA',
    'LFGM3': 'OppFGM3',
    'LFGA3': 'OppFGA3',
    'LFTM': 'OppFTM',
    'LFTA': 'OppFTA',
    'LOR': 'OppOR',
    'LDR': 'OppDR',
    'LAst': 'OppAst',
    'LTO': 'OppTO',
    'LStl': 'OppStl',
    'LBlk': 'OppBlk',
    'LPF': 'OppPF'
}, inplace=True)

# When a team loses, opponent stats come from the winner columns.
losers_opponents = m_reg[[
    'Season', 'LTeamID', 'WScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3',
    'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF'
]].copy()
losers_opponents.rename(columns={
    'LTeamID': 'TeamID',
    'WScore': 'OppScore',
    'WFGM': 'OppFGM',
    'WFGA': 'OppFGA',
    'WFGM3': 'OppFGM3',
    'WFGA3': 'OppFGA3',
    'WFTM': 'OppFTM',
    'WFTA': 'OppFTA',
    'WOR': 'OppOR',
    'WDR': 'OppDR',
    'WAst': 'OppAst',
    'WTO': 'OppTO',
    'WStl': 'OppStl',
    'WBlk': 'OppBlk',
    'WPF': 'OppPF'
}, inplace=True)

# Combine these opponent stats
opp_stats_df = pd.concat([winners_opponents, losers_opponents], axis=0)

# Aggregate opponent stats per (Season, TeamID) using the mean
opp_agg = opp_stats_df.groupby(['Season', 'TeamID'], as_index=False).agg({
    'OppScore': 'mean',
    'OppFGM': 'mean',
    'OppFGA': 'mean',
    'OppFGM3': 'mean',
    'OppFGA3': 'mean',
    'OppFTM': 'mean',
    'OppFTA': 'mean',
    'OppOR': 'mean',
    'OppDR': 'mean',
    'OppAst': 'mean',
    'OppTO': 'mean',
    'OppStl': 'mean',
    'OppBlk': 'mean',
    'OppPF': 'mean'
})

# Merge the opponent stats with your team features DataFrame
team_features = pd.merge(team_features, opp_agg, on=['Season', 'TeamID'], how='left')

# 1. Scoring Margin: Team's average score minus opponent's average score.
team_features['ScoringMargin'] = team_features['AvgScore'] - team_features['OppScore']

# 2. Field Goal Percentage Differential:
#    First compute FG% for both team and opponent, then take the difference.
team_features['FG%'] = (team_features['AvgFGM'] / team_features['AvgFGA']) * 100
team_features['OppFG%'] = (team_features['OppFGM'] / team_features['OppFGA']) * 100
team_features['FGDiff'] = team_features['FG%'] - team_features['OppFG%']

# 3. Rebound Margin: (AvgOR + AvgDR) minus (OppOR + OppDR)
team_features['Rebounds'] = team_features['AvgOR'] + team_features['AvgDR']
team_features['OppRebounds'] = team_features['OppOR'] + team_features['OppDR']
team_features['ReboundMargin'] = team_features['Rebounds'] - team_features['OppRebounds']

# 4. Turnover Margin: AvgTO minus OppTO
team_features['TurnoverMargin'] = team_features['AvgTO'] - team_features['OppTO']

# 5. R+T Formula: (Rebound Margin * 2) + (AvgStl * 0.5) + (6 - OppStl) + TurnoverMargin, rounded to one digit.
team_features['RT'] = ((team_features['ReboundMargin'] * 2) +
                       (team_features['AvgStl'] * 0.5) +
                       (6 - team_features['OppStl']) +
                       team_features['TurnoverMargin']).round(1)

team_features['HighScoringMargin'] = (team_features['ScoringMargin'] >= 8).astype(int)
team_features['HighFGDiff'] = (team_features['FGDiff'] >= 7.5).astype(int)
team_features['HighReboundMargin'] = (team_features['ReboundMargin'] >= 5).astype(int)
team_features['PositiveTurnoverMargin'] = (team_features['TurnoverMargin'] > 0).astype(int)

# Example function: build matchup-level features from two teams' feature Series.
def build_features_for_matchup(teamA_features, teamB_features):
    features = {}
    # Basic features:
    basic_features = ['AvgScore', 'WinPct']
    for feature in basic_features:
        features[f"{feature}_diff"] = teamA_features[feature] - teamB_features[feature]
        features[f"{feature}_ratio"] = teamA_features[feature] / (teamB_features[feature] + 1e-9)
    
    # Advanced metrics:
    advanced_features = ['ScoringMargin', 'FGDiff', 'ReboundMargin', 'TurnoverMargin', 'RT']
    for feature in advanced_features:
        features[f"{feature}_diff"] = teamA_features[feature] - teamB_features[feature]
        features[f"{feature}_ratio"] = teamA_features[feature] / (teamB_features[feature] + 1e-9)
    
    # Binary flags (if any):
    binary_features = ['HighScoringMargin', 'HighFGDiff', 'HighReboundMargin', 'PositiveTurnoverMargin']
    for feature in binary_features:
        features[f"{feature}"] = teamA_features[feature]
        features[f"{feature}_diff"] = teamA_features[feature] - teamB_features[feature]
    
    return features

train_rows = []

# Iterate over tournament games from tourney_results
for idx, game in tourney_results.iterrows():
    season = game['Season']
    wteam = game['WTeamID']
    lteam = game['LTeamID']
    
    # Retrieve team features for winning and losing teams
    w_feat_rows = team_features[(team_features['Season'] == season) & (team_features['TeamID'] == wteam)]
    l_feat_rows = team_features[(team_features['Season'] == season) & (team_features['TeamID'] == lteam)]
    
    # Skip game if features are missing
    if w_feat_rows.empty or l_feat_rows.empty:
        continue
    
    w_features = w_feat_rows.iloc[0]
    l_features = l_feat_rows.iloc[0]
    
    # Create matchup with winning team as TeamA (label=1)
    row_features = build_features_for_matchup(w_features, l_features)
    row_features['TeamA_ID'] = wteam
    row_features['TeamB_ID'] = lteam
    train_rows.append((row_features, 1))
    
    # Optionally, add reversed matchup (losing team as TeamA, label=0)
    row_features_rev = build_features_for_matchup(l_features, w_features)
    row_features_rev['TeamA_ID'] = lteam
    row_features_rev['TeamB_ID'] = wteam
    train_rows.append((row_features_rev, 0))

# Build your feature matrix X and target vector y
X = pd.DataFrame([r[0] for r in train_rows])
y = pd.Series([r[1] for r in train_rows])

from sklearn.model_selection import train_test_split

# Split data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training set size:", X_train.shape, " Test set size:", X_test.shape)

# --------------------------
# Model Training and Evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

# Logistic Regression
log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_proba_lr = log_reg.predict_proba(X_test)
y_pred_lr = log_reg.predict(X_test)
logloss_lr = log_loss(y_test, y_pred_proba_lr)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression - Log Loss: {logloss_lr:.4f}, Accuracy: {accuracy_lr:.4f}")

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_proba_rf = rf_model.predict_proba(X_test)
y_pred_rf = rf_model.predict(X_test)
logloss_rf = log_loss(y_test, y_pred_proba_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest - Log Loss: {logloss_rf:.4f}, Accuracy: {accuracy_rf:.4f}")

from sklearn.neighbors import KNeighborsClassifier

# k-Nearest Neighbors (kNN) Classifier
knn_model = KNeighborsClassifier(n_neighbors=10, metric='manhattan')
knn_model.fit(X_train, y_train)

from sklearn.svm import SVC

# Support Vector Machine (SVM) Classifier
svm_model = SVC(probability=True, kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

from sklearn.ensemble import VotingClassifier

# --------------------------
# Ensemble Model using VotingClassifier (Soft Voting)
ensemble_model = VotingClassifier(
    estimators=[('lr', log_reg), ('rf', rf_model), ('knn', knn_model), ('svm', svm_model), ('gb', gb_model)],
    voting='soft'
)
ensemble_model.fit(X_train, y_train)

y_pred_proba_ensemble = ensemble_model.predict_proba(X_test)
y_pred_ensemble = ensemble_model.predict(X_test)

logloss_ensemble = log_loss(y_test, y_pred_proba_ensemble)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f"Ensemble (Soft Voting) - Log Loss: {logloss_ensemble:.4f}, Accuracy: {accuracy_ensemble:.4f}")

def predict_game(team1, team2, year, model, features_df):
    team1_features = features_df[(features_df['Season'] == year) & (features_df['TeamID'] == team1)]
    team2_features = features_df[(features_df['Season'] == year) & (features_df['TeamID'] == team2)]
    if team1_features.empty or team2_features.empty:
        raise ValueError('Missing team features for one of the teams.')
    team1_features = team1_features.iloc[0]
    team2_features = team2_features.iloc[0]
    matchup_features = build_features_for_matchup(team1_features, team2_features)
    # Add TeamA_ID and TeamB_ID to match training feature names
    matchup_features['TeamA_ID'] = team1
    matchup_features['TeamB_ID'] = team2
    df = pd.DataFrame([matchup_features])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    return pred, proba

def simulate_bracket(south, midwest, east, west, year, features_df, model):
    def simulate_region(region):
        teams = region.copy()
        round_num = 1
        results = []
        while len(teams) > 1:
            new_teams = []
            for i in range(0, len(teams), 2):
                team1 = teams[i]
                team2 = teams[i+1]
                pred, proba = predict_game(team1, team2, year, model, features_df)
                if pred == 1:
                    winner = team1
                    confidence = proba[1]
                else:
                    winner = team2
                    confidence = proba[0]
                results.append((round_num, team1, team2, winner, confidence))
                new_teams.append(winner)
            teams = new_teams
            round_num += 1
        return teams[0], results

    winners = {}
    region_results = {}
    for region, name in zip([south, midwest, east, west], ["South", "Midwest", "East", "West"]):
        winner, results = simulate_region(region)
        winners[name] = winner
        region_results[name] = results

    # Final Four matchups
    ff_results = []
    pred, proba = predict_game(winners["South"], winners["East"], year, model, features_df)
    if pred == 1:
        finalist1 = winners["South"]
        confidence1 = proba[1]
    else:
        finalist1 = winners["East"]
        confidence1 = proba[0]
    ff_results.append(("South vs East", winners["South"], winners["East"], finalist1, confidence1))
    
    pred, proba = predict_game(winners["Midwest"], winners["West"], year, model, features_df)
    if pred == 1:
        finalist2 = winners["Midwest"]
        confidence2 = proba[1]
    else:
        finalist2 = winners["West"]
        confidence2 = proba[0]
    ff_results.append(("Midwest vs West", winners["Midwest"], winners["West"], finalist2, confidence2))
    
    # Championship game
    pred, proba = predict_game(finalist1, finalist2, year, model, features_df)
    if pred == 1:
        champion = finalist1
        champ_conf = proba[1]
    else:
        champion = finalist2
        champ_conf = proba[0]
    champ_results = [("Championship", finalist1, finalist2, champion, champ_conf)]
    
    return winners, region_results, ff_results, champ_results

# Define region seed arrays for last year's tournament (adjust these IDs as needed)
south = [1120, 1106, 1257, 1166, 1276, 1471, 1401, 1463, 1279, 1314, 1235, 1252, 1266, 1307, 1277, 1136]
midwest = [1222, 1188, 1211, 1208, 1155, 1270, 1345, 1219, 1228, 1462, 1246, 1407, 1417, 1429, 1397, 1459]
east = [1181, 1110, 1280, 1124, 1332, 1251, 1112, 1103, 1140, 1433, 1458, 1285, 1388, 1435, 1104, 1352]
west = [1196, 1313, 1163, 1328, 1272, 1161, 1268, 1213, 1281, 1179, 1403, 1423, 1242, 1116, 1385, 1303]

# Set the season year for last year's tournament (e.g., 2023)
last_year = 2025

# Run the bracket simulation using team_features as the features DataFrame and ensemble_model as the model
winners, region_results, ff_results, champ_results = simulate_bracket(south, midwest, east, west, last_year, team_features, ensemble_model)

print("Regional Winners:")
print(winners)
print("\nRegion Results:")
for region, results in region_results.items():
    print(region, results)
print("\nFinal Four Results:")
print(ff_results)
print("\nChampionship Results:")
print(champ_results)