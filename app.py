from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)

# =============================================
# Train model on startup
# =============================================
def add_features(df):
    df = df.copy()
    df['temp_diff']     = df['Process temperature [K]'] - df['Air temperature [K]']
    df['power_proxy']   = df['Torque [Nm]'] * df['Rotational speed [rpm]'] / 1000
    df['wear_torque']   = df['Tool wear [min]'] * df['Torque [Nm]'] / 100
    df['rpm_excess']    = np.maximum(0, df['Rotational speed [rpm]'] - 2000)
    df['torque_excess'] = np.maximum(0, df['Torque [Nm]'] - 60)
    df['wear_excess']   = np.maximum(0, df['Tool wear [min]'] - 200)
    return df

FEATURES = [
    'Type', 'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
    'temp_diff', 'power_proxy', 'wear_torque',
    'rpm_excess', 'torque_excess', 'wear_excess'
]

print("Loading model...")
df = pd.read_csv('ai4i2020.csv')
df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})
df = add_features(df)

model = RandomForestClassifier(
    n_estimators=200, random_state=42,
    class_weight='balanced', max_depth=12, min_samples_leaf=5
)
model.fit(df[FEATURES], df['Machine failure'])
print("Model ready!")

# =============================================
# Routes
# =============================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    row = pd.DataFrame([{
        'Type':                      int(data['type']),
        'Air temperature [K]':       float(data['air']),
        'Process temperature [K]':   float(data['proc']),
        'Rotational speed [rpm]':    float(data['rpm']),
        'Torque [Nm]':               float(data['torque']),
        'Tool wear [min]':           float(data['wear']),
    }])
    row = add_features(row)
    prob = model.predict_proba(row[FEATURES])[0][1]
    return jsonify({'probability': round(float(prob), 4)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
