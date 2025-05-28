import os
import cv2
import numpy as np
import pickle
import base64
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import io
from PIL import Image

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

def preprocess_image(img_bytes):
    """
    Preprocess an image from bytes data to a normalized 28x28 array
    suitable for MNIST model inference.

    Args:
        img_bytes: Image data as bytes

    Returns:
        tuple: (model_input, original_base64, preprocessed_base64)
            - model_input: numpy array of shape (1, 784) for model input
            - original_base64: base64 encoding of the original image
            - preprocessed_base64: base64 encoding of the preprocessed 28x28 image
    """
    # Open image from bytes data
    image = Image.open(io.BytesIO(img_bytes))

    # Convert to OpenCV format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Create a copy for the original thumbnail
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

def extract_base64_data(data_uri):
    """
    Extract the base64 encoded binary data from a data URI.

    Args:
        data_uri: Data URI string starting with "data:image/..."

    Returns:
        bytes: Decoded image data as bytes
    """
    # Extract the base64 part from the data URI
    header, encoded = data_uri.split(",", 1)
    image_data = base64.b64decode(encoded)
    return image_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data received'}), 400

    img_bytes = None

    if 'file' in data:
        # Handle file upload
        try:
            img_bytes = extract_base64_data(data['file'])
        except Exception as e:
            return jsonify({'error': f'Invalid file data: {str(e)}'}), 400
    elif 'canvas' in data:
        # Handle canvas drawing
        try:
            img_bytes = extract_base64_data(data['canvas'])
        except Exception as e:
            return jsonify({'error': f'Invalid canvas data: {str(e)}'}), 400
    else:
        return jsonify({'error': 'No file or canvas data provided'}), 400

    try:
        # Preprocess the image and get base64 encodings
        processed_img, original_b64, processed_b64 = preprocess_image(img_bytes)

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

        return jsonify({
            'prediction': int(prediction),
            'probs': probabilities.tolist(),
            'original': f'data:image/png;base64,{original_b64}',
            'preprocessed': f'data:image/png;base64,{processed_b64}'
        })
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
