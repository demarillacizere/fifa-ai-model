from flask import Flask, request,  render_template
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)

# load the trained model
model = joblib.load("model.joblib")

@app.route('/')
def index():
    return render_template('index.html')



UPLOAD_FOLDER = 'uploads'

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return render_template('index.html', message='No file uploaded')

    file = request.files['file']

    # Check if the file is a CSV
    if not file.filename.endswith('.csv'):
        return render_template('index.html', message='Only CSV files are allowed')
    
    file = request.files['file']
    columns =['age', 'attacking_crossing', 'attacking_short_passing', 'dribbling', 'international_reputation', 'mentality_aggression', 'mentality_composure', 'mentality_vision', 'movement_reactions', 'passing', 'physic', 'potential', 'power_long_shots', 'power_shot_power', 'release_clause_eur', 'shooting', 'skill_ball_control', 'skill_curve', 'skill_long_passing', 'value_eur', 'wage_eur']
    df = pd.read_csv(file)

    df.drop(['sofifa_id', 'player_url', 'club_name', 'nationality_name', 'long_name', 'real_face', 'nation_team_id', 'player_tags', 'player_traits', 'player_face_url', 'club_logo_url', 'club_flag_url', 'nation_logo_url', 'nation_flag_url', 'club_joined'], axis=1, inplace=True)

    # Converting string columns to numerical using label encoding
    le = LabelEncoder()
    df['preferred_foot'] = le.fit_transform(df['preferred_foot'])
    df['work_rate'] = le.fit_transform(df['work_rate'])
    df['body_type'] = le.fit_transform(df['body_type'])
    df['club_position'] = le.fit_transform(df['club_position'].fillna('NaN'))
    df['player_positions'] = le.fit_transform(df['player_positions'])
    df['league_name'] = le.fit_transform(df['league_name'])
    df['dob'] = le.fit_transform(df['dob'])
    # df['short_name'] = le.fit_transform(df['short_name'])

    # Filling empty values with mean or mode
    df['release_clause_eur'].fillna(df['release_clause_eur'].mean(), inplace=True)
    df['height_cm'].fillna(df['height_cm'].mean(), inplace=True)
    df['weight_kg'].fillna(df['weight_kg'].mean(), inplace=True)
    df['pace'].fillna(df['pace'].mode()[0], inplace=True)
    df['shooting'].fillna(df['shooting'].mode()[0], inplace=True)
    df['passing'].fillna(df['passing'].mode()[0], inplace=True)
    df['dribbling'].fillna(df['dribbling'].mode()[0], inplace=True)
    df['defending'].fillna(df['defending'].mode()[0], inplace=True)
    df['physic'].fillna(df['physic'].mode()[0], inplace=True)
    
    df.fillna(df.mean(numeric_only=True).round(1), inplace=True)

    y_pred = model.predict(df[columns])
    y_pred = np.round(y_pred).astype(int)
    realvalues = df['overall'].iloc[df.index]
    names = df['short_name'].iloc[df.index]
    return render_template('index.html', predictions=list(y_pred)[:10],realvalues=list(realvalues)[:10],names=list(names)[:10])

if __name__ == '__main__':
    app.run(debug=True)