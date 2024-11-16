from flask import Flask, request, jsonify
import pickle
from utils import drop_features

def drop_features(X, df):
    return X.drop(columns=df, axis=1)

# Loading the model
model = pickle.load(open('models/xgb_model.pkl', 'rb'))

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    features = data['features']
    # Make a prediction
    prediction = model.predict([features])
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
   app.run(debug=True)
