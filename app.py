import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load models
rf_model_d1 = pickle.load(open('rf_model_d1.pkl', 'rb'))
rf_model_d2 = pickle.load(open('rf_model_d2.pkl', 'rb'))
rf_model_d3 = pickle.load(open('rf_model_d3.pkl', 'rb'))
rf_model_d6 = pickle.load(open('rf_model_d6.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input features from the form
        int_features = [float(x) for x in request.form.values()]
        features = [np.array(int_features)]

        # Predict using each model
        prediction_d1 = rf_model_d1.predict(features)[0]
        prediction_d2 = rf_model_d2.predict(features)[0]
        prediction_d3 = rf_model_d3.predict(features)[0]
        prediction_d6 = rf_model_d6.predict(features)[0]

        # Format the prediction output
        output = f'D1: {round(prediction_d1, 2)}, D2: {round(prediction_d2, 2)}, D3: {round(prediction_d3, 2)}, D6: {round(prediction_d6, 2)}'

        return render_template('index.html', prediction_text=f'Retention predictions: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


# if __name__ == "__main__":
#     app.run(debug=True)
