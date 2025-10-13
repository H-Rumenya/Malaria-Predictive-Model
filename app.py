from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

app = Flask(__name__)

# Define model directory and timestamp
MODEL_DIR = 'Models'
TIMESTAMP = '20251008_180851'  # Replace with the actual timestamp from the notebook run (e.g., '20251008_XXXXXX')

# Load regression model and metadata
reg_metadata_path = os.path.join(MODEL_DIR, f'regression_metadata_{TIMESTAMP}.json')
with open(reg_metadata_path, 'r') as f:
    reg_metadata = json.load(f)
reg_model_path = os.path.join(MODEL_DIR, f'regression_model_{TIMESTAMP}.pkl')
reg_model = joblib.load(reg_model_path)
reg_features = reg_metadata['features']

# Load classification model and metadata
class_metadata_path = os.path.join(MODEL_DIR, f'classification_metadata_{TIMESTAMP}.json')
with open(class_metadata_path, 'r') as f:
    class_metadata = json.load(f)
class_model_path = os.path.join(MODEL_DIR, f'classification_model_{TIMESTAMP}.pkl')
class_dict = joblib.load(class_model_path)
class_model = class_dict['model']
class_threshold = class_dict['threshold']
class_features = class_metadata['features']

# Ensure features match
if set(reg_features) != set(class_features):
    raise ValueError("Feature mismatch between regression and classification models!")
ALL_FEATURES = reg_features

@app.route('/', methods=['GET'])
def home():
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kakuma Malaria Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            min-height: 100vh;
            color: #333;
        }
        .container {
            background: #ffffff;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
            width: 95%;
            max-width: 900px;
            margin: 20px;
            animation: fadeIn 1s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header img {
            max-width: 100px;
            margin-bottom: 10px;
        }
        h1 {
            color: #1565c0;
            font-weight: 700;
            margin: 0;
            font-size: 2.5em;
        }
        p {
            color: #555;
            font-size: 1.1em;
            margin: 10px 0 20px;
            text-align: center;
        }
        form {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .top-inputs {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .input-group {
            position: relative;
        }
        label {
            display: block;
            font-weight: 500;
            color: #333;
            margin-bottom: 5px;
            font-size: 1em;
        }
        input[type="number"], select {
            width: 100%;
            padding: 12px;
            border: 1px solid #ccc;
            border-radius: 8px;
            font-size: 1em;
            transition: all 0.3s ease;
            box-sizing: border-box;
        }
        input[type="number"]:focus, select:focus {
            border-color: #1565c0;
            box-shadow: 0 0 8px rgba(21, 101, 192, 0.3);
            outline: none;
        }
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #333;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 8px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.9em;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        .week-box {
            background: #f5faff;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #e0e7ff;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        .week-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 20px rgba(21, 101, 192, 0.2);
        }
        h3 {
            color: #0d47a1;
            font-size: 1.4em;
            margin-bottom: 15px;
            border-bottom: 2px solid #42a5f5;
            padding-bottom: 5px;
        }
        .submit-btn {
            background: #1565c0;
            color: white;
            padding: 15px;
            border: none;
            border-radius: 10px;
            font-size: 1.2em;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .submit-btn:hover {
            background: #0d47a1;
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(21, 101, 192, 0.4);
        }
        .submit-btn::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            transition: width 0.6s ease, height 0.6s ease;
        }
        .submit-btn:active::after {
            width: 300px;
            height: 300px;
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #fff;
            background: #1565c0;
            width: 100%;
            font-size: 1em;
            position: relative;
            bottom: 0;
        }
        @media (max-width: 600px) {
            .container {
                padding: 20px;
            }
            h1 {
                font-size: 1.8em;
            }
            .top-inputs {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Kakuma Malaria Predictor</h1>
            <p>Predict Malaria Cases or Detect outbreak (next week's cases ≥ 2000 and ≥ 1.5 times current week's cases) using weekly weather data.</p>
        </div>

        <form action="/submit" method="POST">
            <div class="top-inputs">
                <div class="input-group">
                    <label class="tooltip">Number of Weeks (1–4):
                        <span class="tooltiptext">Select how many weeks to predict (1 to 4).</span>
                    </label>
                    <input type="number" name="period" min="1" max="4" value="4" required>
                </div>
                <div class="input-group">
                    <label class="tooltip">Task:
                        <span class="tooltiptext">Choose to predict case counts, detect outbreak, or both.</span>
                    </label>
                    <select name="task" required>
                        <option value="regression">Regression (Case Counts)</option>
                        <option value="classification">Classification (Outbreak Detection)</option>
                        <option value="both">Both (Regression + Classification)</option>
                    </select>
                </div>
                <div class="input-group">
                    <label class="tooltip">Current Malaria Cases:
                        <span class="tooltiptext">Enter current week's malaria cases (default: 0).</span>
                    </label>
                    <input type="number" name="Combined_positive" step="1" value="0" required>
                </div>
            </div>

            {% for i in range(4) %}
            <div class="week-box">
                <h3>Week {{ i+1 }}</h3>
                <div class="input-group">
                    <label class="tooltip">Temperature (°C):
                        <span class="tooltiptext">Enter temperature in Celsius (e.g., 25.1234).</span>
                    </label>
                    <input type="number" name="temp_c_{{ i }}" step="0.0001" placeholder="25.1234">
                </div>
                <div class="input-group">
                    <label class="tooltip">Relative Humidity (%):
                        <span class="tooltiptext">Enter humidity percentage (e.g., 60.1234).</span>
                    </label>
                    <input type="number" name="rh_pct_{{ i }}" step="0.0001" placeholder="60.1234">
                </div>
                <div class="input-group">
                    <label class="tooltip">Rainfall (mm):
                        <span class="tooltiptext">Enter rainfall in millimeters (e.g., 10.1234).</span>
                    </label>
                    <input type="number" name="rain_mm_{{ i }}" step="0.0001" placeholder="10.1234">
                </div>
                <div class="input-group">
                    <label class="tooltip">Wind Speed (km/h):
                        <span class="tooltiptext">Enter wind speed in km/h (e.g., 5.1234).</span>
                    </label>
                    <input type="number" name="wind10_kmh_{{ i }}" step="0.0001" placeholder="5.1234">
                </div>
                <div class="input-group">
                    <label class="tooltip">Soil Moisture (m³/m³):
                        <span class="tooltiptext">Enter soil moisture in m³/m³ (e.g., 0.1234).</span>
                    </label>
                    <input type="number" name="soil_moisture_top_m3m3_{{ i }}" step="0.0001" placeholder="0.1234">
                </div>
            </div>
            {% endfor %}

            <button type="submit" class="submit-btn">Predict</button>
        </form>
    </div>

    <footer>© 2025 Group 2 Kakuma Malaria Predictor</footer>
</body>
</html>
    ''')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        period = int(request.form.get('period', 4))
        task = request.form.get('task', 'regression').lower()

        if period < 1 or period > 4:
            return "Error: Period must be between 1 and 4.", 400
        if task not in ['regression', 'classification', 'both']:
            return "Error: Task must be 'regression', 'classification', or 'both'.", 400

        period_data = []
        combined_positive = float(request.form.get('Combined_positive', 0))
        for i in range(period):
            row = {
                'temp_c': float(request.form.get(f'temp_c_{i}', 0)),
                'rh_pct': float(request.form.get(f'rh_pct_{i}', 0)),
                'rain_mm': float(request.form.get(f'rain_mm_{i}', 0)),
                'wind10_kmh': float(request.form.get(f'wind10_kmh_{i}', 0)),
                'soil_moisture_top_m3m3': float(request.form.get(f'soil_moisture_top_m3m3_{i}', 0)),
                'Combined_positive': combined_positive if i == 0 else 0
            }
            period_data.append(row)

        start_date = datetime(2025, 10, 4)
        week_starts = [start_date + timedelta(weeks=i) for i in range(period)]

        df_list = []
        for i, (week_start, row_data) in enumerate(zip(week_starts, period_data)):
            row = {'week_start': week_start}
            for key in ['temp_c', 'rh_pct', 'rain_mm', 'wind10_kmh', 'soil_moisture_top_m3m3']:
                row[key] = row_data.get(key, 0)
            row['Combined positive'] = row_data.get('Combined_positive', 0)
            df_list.append(row)

        df = pd.DataFrame(df_list)
        df['month'] = df['week_start'].dt.month
        df['week_of_year'] = df['week_start'].dt.isocalendar().week.astype(float)
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['sin_week'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['cos_week'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        df['sin_month_2'] = np.sin(4 * np.pi * df['month'] / 12)
        df['cos_month_2'] = np.cos(4 * np.pi * df['month'] / 12)
        df['ratio'] = 1.0
        df['ratio_lag_1'] = df['ratio'].shift(1).fillna(1.0)
        df['ratio_lag_2'] = df['ratio'].shift(2).fillna(1.0)

        results = {}
        for i in range(len(df)):
            if i > 0 and task in ['regression', 'both'] and 'regression' in results and len(results['regression']) > i-1:
                df.loc[i, 'Combined positive'] = results['regression'][i-1]

            for window in [4, 8, 12]:
                if len(df) >= window:
                    df[f'Combined_positive_roll_mean_{window}'] = df['Combined positive'].shift(1).rolling(window=window, min_periods=1).mean().fillna(0)
                    df[f'Combined_positive_roll_std_{window}'] = df['Combined positive'].shift(1).rolling(window=window, min_periods=1).std().fillna(0)
                else:
                    df[f'Combined_positive_roll_mean_{window}'] = 0
                    df[f'Combined_positive_roll_std_{window}'] = 0

            df['rain_soil_interaction'] = df['rain_mm'] * df['soil_moisture_top_m3m3']
            df['temp_rh_interaction'] = df['temp_c'] * df['rh_pct']
            df['temp_combined_interaction'] = df['temp_c'] * df['Combined positive']

            if i < len(df) - 1 and task in ['regression', 'both'] and 'regression' in results and len(results['regression']) > i:
                df.loc[i, 'ratio'] = results['regression'][i] / df.loc[i, 'Combined positive'] if df.loc[i, 'Combined positive'] != 0 else 1.0
                df['ratio_lag_1'] = df['ratio'].shift(1).fillna(1.0)
                df['ratio_lag_2'] = df['ratio'].shift(2).fillna(1.0)

            X = df[ALL_FEATURES].astype(float)
            if task in ['regression', 'both']:
                reg_preds = reg_model.predict(X.iloc[[i]]).tolist()
                results.setdefault('regression', []).extend(reg_preds)
            if task in ['classification', 'both']:
                class_probs = class_model.predict_proba(X.iloc[[i]])[:, 1]
                class_preds = (class_probs >= class_threshold).astype(int).tolist()
                results.setdefault('classification', []).extend(class_preds)
                results.setdefault('class_probabilities', []).extend(class_probs.tolist())

        html = '''
        <html><head>
        <title>Prediction Results</title>
        <style>
        body {font-family: 'Roboto', sans-serif; margin: 40px; background: #f5faff;}
        .container {max-width: 900px; margin: auto; padding: 20px; background: #fff; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);}
        h2 {color: #1565c0; text-align: center; font-weight: 700;}
        p {color: #555; font-size: 1em; text-align: center;}
        table {border-collapse: collapse; width: 100%; margin: 20px 0;}
        th,td {border: 1px solid #e0e7ff; padding: 12px; text-align: center;}
        th {background: #1565c0; color: white; font-weight: 700;}
        tr:nth-child(even) {background: #f5faff;}
        tr:hover {background: #e3f2fd;}
        .back-btn {display: block; width: 200px; margin: 20px auto; padding: 12px; background: #1565c0; color: white; text-align: center; text-decoration: none; border-radius: 8px; transition: all 0.3s ease;}
        .back-btn:hover {background: #0d47a1; transform: scale(1.05);}
        </style></head>
        <body>
        <div class="container">
            <h2>Prediction Results</h2>
            <p><b>Note:</b> A "Outbreak" indicates an outbreak where next week's malaria cases are predicted to be ≥ 2000 and ≥ 1.5 times the current week's cases.</p>
            <table>
                <tr>
                    <th>Week</th>
                    {% if task in ['regression', 'both'] %}
                    <th>Predicted Cases</th>
                    {% endif %}
                    {% if task in ['classification', 'both'] %}
                    <th>Outbreak Status</th><th>Probability</th>
                    {% endif %}
                </tr>
                {% for i in range(period) %}
                <tr>
                    <td>{{ week_starts[i] }}</td>
                    {% if task in ['regression', 'both'] %}
                    <td>{{ results.regression[i] | round(2) }}</td>
                    {% endif %}
                    {% if task in ['classification', 'both'] %}
                    <td>{{ 'Malaria outbreak' if results.classification[i] == 1 else 'No Malaria Outbreak' }}</td>
                    <td>{{ results.class_probabilities[i] | round(3) }}</td>
                    {% endif %}
                </tr>
                {% endfor %}
            </table>
            <a href="/" class="back-btn">Back to Form</a>
        </div>
        </body></html>
        '''
        return render_template_string(html, period=period, results=results, week_starts=[ws.strftime('%Y-%m-%d') for ws in week_starts], task=task)

    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)