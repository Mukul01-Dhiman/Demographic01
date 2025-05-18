from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

app = Flask(__name__)

# Load dataset
data = pd.read_excel('C://Users//LENOVO//OneDrive//Desktop//Book1.xlsx')

# Clean and prepare data
features = ['TOT_F', 'M_LIT', 'F_LIT', 'TOT_WORK_P', 'MAINWORK_P', 'MARGWORK_P', 'Income', 'Fertility_Rate']
target = 'P_LIT'
data = data.dropna(subset=features + [target])  # Remove rows with missing data

@app.route('/')
def index():
    states = data['State'].unique().tolist()
    return render_template('ind.html', states=states)  # You can create form input fields for each feature

@app.route('/predict', methods=['POST'])
def predict():
    req = request.json

    # Extract values from frontend JSON
    input_data = [req[feature] for feature in features]

    # Prepare training data
    X = data[features]
    y = data[target]

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    dt_model = DecisionTreeRegressor()
    dt_model.fit(train_X, train_y)

    # Accuracy of model
    y_pred = dt_model.predict(test_X)
    acc = round(r2_score(test_y, y_pred) * 100, 2)

    # Predict
    predicted_lit = round(dt_model.predict([input_data])[0], 2)

    return jsonify({
        'predicted_literacy': predicted_lit,
        'accuracy': acc
        'state': state,
        'year1': year1,
        'year2': year2,
        'predicted_lit_1': round(pred1, 2),
        'predicted_lit_2': round(pred2, 2),
        'change_percent': round(change, 2),
        'male_lit_1': round(male_pred1, 2),
        'male_lit_2': round(male_pred2, 2),
        'female_lit_1': round(female_pred1, 2),
        'female_lit_2': round(female_pred2, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
