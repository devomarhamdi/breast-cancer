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
        if data:
            # Parse JSON data into a Python dictionary
            data_dict = json.loads(data)

            # Extract values and convert to NumPy array
            data_array = np.array(list(data_dict.values()))

            # Reshape the array to have one row and an undetermined number of columns
            data_array_reshaped = data_array.reshape(1, -1)

            prediction_list = model.predict(data_array_reshaped)

            # Return prediction as JSON response
            return jsonify({"prediction": prediction_list})
        else:
            return jsonify({"error": "No data provided"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
