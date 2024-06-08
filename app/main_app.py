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

    try:
        # Read the image file
        img = Image.open(file).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28 pixels
        img_array = np.array(img)

        # Normalize the image array
        img_array = img_array / 255.0

        # Flatten the array to 1D (784,)
        img_array = img_array.reshape(1, -1)  # Reshape for the model

        # Make a prediction
        prediction, pred_proba = model_predict(img_array)

        predicted_class = prediction[0]
        pred_probability = pred_proba[0]


        return jsonify({'predicted_class': int(predicted_class),
                        'pred_probability': float(pred_probability)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
