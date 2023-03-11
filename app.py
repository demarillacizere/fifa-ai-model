from flask import Flask, request,  render_template
import joblib
import pandas as pd
import numpy as np
app = Flask(__name__)

# load the trained model
model = joblib.load("fifa_model.joblib")

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
    columns = ['attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure', 'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']
    data = pd.read_csv(file)
    y_pred = model.predict(data[columns])
    y_pred = np.round(y_pred)
    realvalues = data['overall'].iloc[data.index]

    return render_template('index.html', predictions=list(y_pred)[:10],realvalues=list(realvalues)[:10])

if __name__ == '__main__':
    app.run(debug=True)