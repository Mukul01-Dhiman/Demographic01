from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model1.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form values and convert to float
        features = [float(x) for x in request.form.values()]
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]
        return render_template('index1.html', prediction_text=f'Predicted P_LIT: {prediction:.2f}')
    except Exception as e:
        return render_template('index1.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
