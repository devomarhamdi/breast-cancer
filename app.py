from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pre-trained machine learning model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive data from POST request
        data = request.json
        jsonData = pd.DataFrame(data)
        # Check if data is provided
        if data:
            # Make predictions using the loaded model
            prediction = model.predict(jsonData)
            prediction_series = pd.Series(prediction, index=[0])  # Assuming the prediction is for the first row

            # Convert prediction to list (optional)
            prediction_list = prediction_series.tolist()

            # Return prediction as JSON response
            return jsonify({"prediction": prediction_list})
        else:
            return jsonify({"error": "No data provided"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
