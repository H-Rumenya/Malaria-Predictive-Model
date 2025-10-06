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
TIMESTAMP = '20251005_204205'  # Replace with your actual timestamp from model_training.py

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
    <style>
        body {
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #e3f2fd, #f1f8e9);
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            min-height: 100vh;
        }
        .container {
            background: #eaf4ff; /* soft blue background */
            padding: 30px 40px;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            width: 95%;
            max-width: 850px;
            margin-bottom: 40px;
            animation: fadeIn 0.8s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }
        h1 {
            color: #2e7d32;
            text-align: center;
            margin-bottom: 10px;
        }
        p {
            text-align: center;
            color: #333;
            margin-bottom: 25px;
        }
        form {
            text-align: center;
        }
        label {
            display: inline-block;
            width: 220px;
            text-align: left;
            color: #333;
            font-weight: 500;
        }
        input[type="number"], select {
            width: 160px;
            padding: 7px;
            border-radius: 6px;
            border: 1px solid #ccc;
            font-size: 14px;
            transition: all 0.3s ease;
        }
        input[type="number"]::placeholder {
            color: #aaa;
            opacity: 0.7;
        }
        input[type="number"]:focus, select:focus {
            border-color: #4CAF50;
            box-shadow: 0 0 5px rgba(76,175,80,0.4);
            outline: none;
        }
        .week-box {
            background-color: #f3f9ff; /* light soft blue for each week box */
            border: 1px solid #c8e1ff;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            text-align: left;
            transition: all 0.3s ease-in-out;
        }
        .week-box:hover {
            transform: scale(1.03);
            box-shadow: 0 4px 12px rgba(76,175,80,0.25);
            border-color: #4CAF50;
            background-color: #e1f5fe; /* slightly brighter blue on hover */
        }
        h3 {
            color: #1b5e20;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 5px;
            margin-bottom: 10px;
        }
        .submit-btn {
            width: 100%;
            padding: 12px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .submit-btn:hover {
            background-color: #43a047;
            transform: scale(1.02);
        }
        footer {
            text-align: center;
            padding: 12px;
            color: #333;
            width: 100%;
            font-size: 14px;
        }
        .top-inputs {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            margin-bottom: 25px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Kakuma Malaria Predictor</h1>
        <p>Enter weekly weather data and optionally current malaria cases for up to 4 weeks to predict future cases and detect spikes.</p>

        <form action="/submit" method="POST">
            <div class="top-inputs">
                <div>
                    <label>Number of Weeks (1–4):</label>
                    <input type="number" name="period" min="1" max="4" value="4" required>
                </div>
                <div>
                    <label>Task:</label>
                    <select name="task" required>
                        <option value="both">Both (Regression + Classification)</option>
                        <option value="regression">Regression (Case Counts)</option>
                        <option value="classification">Classification (Spike Detection)</option>
                    </select>
                </div>
            </div>

            {% for i in range(4) %}
            <div class="week-box">
                <h3>Week {{ i+1 }}</h3>
                <label>Temperature (°C):</label>
                <input type="number" name="temp_c_{{ i }}" step="0.1" placeholder="25.5"><br>
                <label>Relative Humidity (%):</label>
                <input type="number" name="rh_pct_{{ i }}" step="0.1" placeholder="60"><br>
                <label>Rainfall (mm):</label>
                <input type="number" name="rain_mm_{{ i }}" step="0.1" placeholder="10"><br>
                <label>Wind Speed (km/h):</label>
                <input type="number" name="wind10_kmh_{{ i }}" step="0.1" placeholder="5"><br>
                <label>Soil Moisture (m³/m³):</label>
                <input type="number" name="soil_moisture_top_m3m3_{{ i }}" step="0.001" placeholder="0.1"><br>
                <label>Current Malaria Cases (optional):</label>
                <input type="number" name="Combined_positive_{{ i }}" step="1" placeholder="Leave blank if unknown"><br>
            </div>
            {% endfor %}

            <button type="submit" class="submit-btn">Predict</button>
        </form>
    </div>

    <footer>© 2025 Group 2 Kakuma Malaria Predictor</footer>
</body>
</html>
    ''')


# ---------- SUBMIT ROUTE  ----------
@app.route('/submit', methods=['POST'])
def submit():
    try:
        period = int(request.form.get('period', 4))
        task = request.form.get('task', 'both').lower()

        if period < 1 or period > 4:
            return "Error: Period must be between 1 and 4.", 400

        period_data = []
        for i in range(period):
            row = {
                'temp_c': float(request.form.get(f'temp_c_{i}', 0)),
                'rh_pct': float(request.form.get(f'rh_pct_{i}', 0)),
                'rain_mm': float(request.form.get(f'rain_mm_{i}', 0)),
                'wind10_kmh': float(request.form.get(f'wind10_kmh_{i}', 0)),
                'soil_moisture_top_m3m3': float(request.form.get(f'soil_moisture_top_m3m3_{i}', 0)),
                'Combined_positive': float(request.form.get(f'Combined_positive_{i}', np.nan))
            }
            period_data.append(row)

        start_date = datetime(2025, 10, 4)
        week_starts = [start_date + timedelta(weeks=i) for i in range(period)]

        df_list = []
        for i, (week_start, row_data) in enumerate(zip(week_starts, period_data)):
            row = {'week_start': week_start}
            for key in ['temp_c', 'rh_pct', 'rain_mm', 'wind10_kmh', 'soil_moisture_top_m3m3']:
                row[key] = row_data.get(key, 0)
            row['Combined positive'] = row_data.get('Combined_positive', np.nan)
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
            if i == 0:
                df.loc[i, 'Combined positive'] = df.loc[i, 'Combined positive'] if not pd.isna(df.loc[i, 'Combined positive']) else 0
            else:
                if task in ['regression', 'both'] and 'regression' in results and len(results['regression']) > i-1:
                    df.loc[i, 'Combined positive'] = results['regression'][i-1]
                else:
                    df.loc[i, 'Combined positive'] = 0

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
        <title>Results</title>
        <style>
        body {font-family: Arial; margin: 20px;}
        table {border-collapse: collapse; width: 100%; max-width: 800px;}
        th,td {border:1px solid #ddd; padding:8px;}
        th {background-color:#4CAF50; color:white;}
        tr:nth-child(even){background:#f2f2f2;}
        </style></head>
        <body>
        <h2>Prediction Results</h2>
        <table><tr><th>Week</th><th>Predicted Cases</th><th>Spike?</th><th>Probability</th></tr>
        {% for i in range(period) %}
        <tr>
        <td>{{ week_starts[i] }}</td>
        <td>{{ results.regression[i] | round(2) if 'regression' in results }}</td>
        <td>{{ results.classification[i] if 'classification' in results }}</td>
        <td>{{ results.class_probabilities[i] | round(3) if 'class_probabilities' in results }}</td>
        </tr>
        {% endfor %}
        </table>
        <br><a href="/">Back to Form</a>
        </body></html>
        '''
        return render_template_string(html, period=period, results=results, week_starts=[ws.strftime('%Y-%m-%d') for ws in week_starts])

    except Exception as e:
        return f"Error: {str(e)}", 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
