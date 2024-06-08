from src.predict import model_predict
from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image

import io

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # Read the image file
    img = Image.open(file).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img_array = np.array(img)

    # Normalize the image array
    img_array = img_array / 255.0
    img_array = img_array.reshape(-1, -1)  # Reshape for the model

    # Make a prediction
    prediction = model_predict(img_array)
    predicted_class = prediction[0]


    return jsonify({'Model Prediction': int(predicted_class)})


if __name__ == '__main__':
    app.run(debug=True)
