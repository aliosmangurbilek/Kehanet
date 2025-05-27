import os
import cv2
import numpy as np
import pickle
import base64
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

model_path = '/home/ali/PycharmProjects/Kehanet/models/trained_mnist_model.pkl'
try:
    model = load_trained_model(model_path)
    print(f"Trained model loaded from {model_path}")
except FileNotFoundError:
    # Try the example folder path if the models folder path fails
    model_path = '/home/ali/PycharmProjects/Kehanet/examples/trained_mnist_model.pkl'
    try:
        model = load_trained_model(model_path)
        print(f"Trained model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Trained model not found at any expected path")

def preprocess(img):
    # Create a copy of the original image for thumbnail
    original = img.copy()

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

    # Save a copy of the processed image for display
    processed_display = (norm * 255).astype(np.uint8)

    # Create base64 strings for both images
    _, thumbnail_buffer = cv2.imencode('.png', cv2.resize(original, (150, 150), interpolation=cv2.INTER_AREA))
    thumbnail_b64 = base64.b64encode(thumbnail_buffer).decode('utf-8')

    _, processed_buffer = cv2.imencode('.png', processed_display)
    processed_b64 = base64.b64encode(processed_buffer).decode('utf-8')

    # Flatten to match model input shape (1, 784)
    model_input = norm.flatten().reshape(1, -1)

    return model_input, thumbnail_b64, processed_b64

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

        # Preprocess the image and get base64 encodings
        processed_img, thumbnail_b64, processed_b64 = preprocess(img)

        # Import Tensor class for prediction
        from core.tensor import Tensor

        # Convert to Tensor for model input
        tensor_input = Tensor(processed_img, requires_grad=False)

        # Make prediction
        output = model(tensor_input)

        # Get the data from tensor and convert to probabilities
        logits = output.data.flatten()

        # Convert logits to probabilities using softmax
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        probabilities = exp_logits / np.sum(exp_logits)

        # Get the predicted digit
        prediction = np.argmax(probabilities)

        # Clean up by removing the uploaded file
        os.remove(filepath)

        return jsonify({
            'prediction': int(prediction),
            'probabilities': probabilities.tolist(),
            'original_image': f'data:image/png;base64,{thumbnail_b64}',
            'processed_image': f'data:image/png;base64,{processed_b64}'
        })

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
