from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and data
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load Excel data
data = pd.read_excel('C://Users//LENOVO//OneDrive//Desktop//Book1.xlsx')  # Make sure this file exists in same folder or give full path

@app.route('/')
def index():
    states = data['State'].unique().tolist()
    return render_template('ind.html', states=states)

@app.route('/predict', methods=['POST'])
def predict():
    req = request.form
    state = req['state']
    year = int(req['year'])

    state_data = data[(data['State'] == state)]
    if state_data.empty:
        return jsonify({'error': 'State not found'}), 400

    # Extract input features
    latest = state_data[state_data['Year'] == state_data['Year'].max()].iloc[0]
    input_features = [
        latest['TOT_F'], latest['M_LIT'], latest['F_LIT'],
        latest['TOT_WORK_P'], latest['MAINWORK_P'],
        latest['MARGWORK_P'], latest['Income'], latest['Fertility_Rate']
    ]

    # Predict
    pred = model.predict([input_features])[0]

    return render_template('ind.html', prediction_text=f"Predicted Literacy Rate for {state} in {year}: {round(pred, 2)}%", states=data['State'].unique().tolist())

if __name__ == '__main__':
    app.run(debug=True)
