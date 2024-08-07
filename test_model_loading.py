import pickle
import numpy as np

# Load models
rf_model_d1 = pickle.load(open('rf_model_d1.pkl', 'rb'))
rf_model_d2 = pickle.load(open('rf_model_d2.pkl', 'rb'))
rf_model_d3 = pickle.load(open('rf_model_d3.pkl', 'rb'))
rf_model_d6 = pickle.load(open('rf_model_d6.pkl', 'rb'))

# Example feature array (replace with actual test data)
test_features = np.array([[8, 4, 0, 3, 4, 2, 11, 11]])

# Predict using the D1 model
model = rf_model_d1
prediction = model.predict(test_features)
print(f'Prediction: {prediction[0]}')
