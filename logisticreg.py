# Step 1: Install Flask and ngrok
!pip install flask-ngrok joblib

# Step 2: Prepare your model
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib



match = pd.read_csv('/content/sample_data/matches.csv')
delivery = pd.read_csv('/content/sample_data/deliveries.csv')

# Preprocess data (same as your q3.py)
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

match['team1'] = match['team1'].replace('Delhi Daredevils', 'Delhi Capitals')
match['team2'] = match['team2'].replace('Delhi Daredevils', 'Delhi Capitals')
match['team1'] = match['team1'].replace('Deccan Chargers', 'Sunrisers Hyderabad')
match['team2'] = match['team2'].replace('Deccan Chargers', 'Sunrisers Hyderabad')

match_df = match[(match['team1'].isin(teams)) & (match['team2'].isin(teams))]
match_df = match_df[match_df['dl_applied'] == 0]
match_df = match_df[['id', 'city', 'winner', 'team1', 'team2']]

delivery_df = delivery.merge(match_df, left_on='match_id', right_on='id')
delivery_df = delivery_df[delivery_df['inning'] == 2]
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: 1 if x != "0" else 0)
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')

first_innings = delivery[delivery['inning'] == 1]
first_innings_score = first_innings.groupby('match_id')['total_runs'].sum().reset_index()
first_innings_score.columns = ['match_id', 'target']
first_innings_score['target'] += 1

delivery_df = delivery_df.merge(first_innings_score, on='match_id')
delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs'].cumsum()
delivery_df['runs_left'] = delivery_df['target'] - delivery_df['current_score']
delivery_df['balls_left'] = 126 - (delivery_df['over'] * 6 + delivery_df['ball'])
delivery_df['wickets'] = 10 - delivery_df.groupby('match_id')['player_dismissed'].cumsum()
delivery_df = delivery_df.dropna(subset=['runs_left', 'balls_left', 'wickets'])
delivery_df['crr'] = (delivery_df['current_score'] * 6) / (120 - delivery_df['balls_left'])
delivery_df['rrr'] = (delivery_df['runs_left'] * 6) / delivery_df['balls_left']

def result(row):
    return 1 if row['winner'] == row['batting_team'] else 0

delivery_df['result'] = delivery_df.apply(result, axis=1)

final_df = delivery_df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'current_score', 'crr', 'rrr', 'result']]
final_df = final_df.sample(frac=1)

X = final_df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'current_score', 'crr', 'rrr']]
y = final_df['result']

# Pipeline
trf = ColumnTransformer([
    ('onehot', OneHotEncoder(drop='first'), ['batting_team', 'bowling_team', 'city'])
])
pipe = Pipeline([
    ('preprocess', trf),
    ('classifier', LogisticRegression(solver='liblinear'))
])
pipe.fit(X, y)

# Save model
joblib.dump(pipe, 'model.pkl')







