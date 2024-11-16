from flask import Flask, request, jsonify
import pickle
import pandas as pd
from utils import *

# Loading the model
model = pickle.load(open('models/xgb_model.pkl', 'rb'))


app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    features = data['features']

    input_data = pd.DataFrame(features, columns=feature_columns)
    # Make a prediction
    prediction = model.predict(input_data)

    print(f"Prediction: {prediction}")
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
   app.run(debug=True)
