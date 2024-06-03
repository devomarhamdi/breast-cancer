from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the pre-trained machine learning model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive data from POST request
        data = request.json
        
        # Convert JSON data into a reshaped NumPy array
        data_array = np.array(list(data.values())).reshape(1, -1)

        # Make predictions using the loaded model
        prediction = model.predict(data_array)
        
        # Assuming your model returns a single prediction, convert it to a list
        prediction = prediction.tolist()
        if (prediction[0] == 0):
            return jsonify({"prediction": 'The Breast cancer is Malignant'})
        else:
            return jsonify({"prediction": 'The Breast Cancer is Benign'})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
