from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)

# Load model
model = load_model("model_final.h5")
class_names = ['Healthy', 'Powdery', 'Rust']

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((150, 150))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img_bytes = file.read()
    img = preprocess_image(img_bytes)
    prediction = model.predict(img)[0]

    index = np.argmax(prediction)
    confidence = float(prediction[index])

    return jsonify({    
        'class': class_names[index],
        'confidence': round(confidence * 100, 2),
        'all_confidences': dict(zip(class_names, map(lambda x: round(x * 100, 2), prediction)))
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
