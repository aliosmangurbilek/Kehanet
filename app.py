import os
import cv2
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load the trained model
def load_trained_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model_path = 'examples/trained_mnist_model.pkl'
try:
    model = load_trained_model(model_path)
    print(f"Trained model loaded from {model_path}")
except FileNotFoundError:
    print(f"Trained model not found at {model_path}")

def preprocess(img):
    # Convert to grayscale if it's a color image
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize to 0-1
    norm = resized.astype(np.float32) / 255.0

    # MNIST expects white digits on black background
    # If the image has black digits on white background, invert it
    if np.mean(norm) > 0.5:  # if background is bright
        norm = 1.0 - norm

    # Flatten to match model input shape (1, 784)
    return norm.flatten().reshape(1, -1)  # Ensures shape is (1, 784)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read image, preprocess and make prediction
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Failed to read image'}), 400

        # Preprocess the image
        processed_img = preprocess(img)

        # Import Tensor class for prediction
        from core.tensor import Tensor

        # Convert to Tensor for model input
        tensor_input = Tensor(processed_img, requires_grad=False)

        # Make prediction
        output = model(tensor_input)

        # Get the data from tensor and find argmax
        prediction = np.argmax(output.data)

        # Clean up by removing the uploaded file
        os.remove(filepath)

        return jsonify({'prediction': int(prediction)})

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
