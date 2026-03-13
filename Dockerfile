FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies required by OpenCV on current Debian slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better cache usage
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create uploads directory if it doesn't exist
RUN mkdir -p uploads

# Fix model path for Docker environment
RUN sed -i 's|/home/ali/PycharmProjects/Kehanet/models/trained_mnist_model.pkl|/app/models/trained_mnist_model.pkl|g' app.py
RUN sed -i 's|/home/ali/PycharmProjects/Kehanet/examples/trained_mnist_model.pkl|/app/examples/trained_mnist_model.pkl|g' app.py

# Expose port for the Flask application
EXPOSE 1314

# Command to run the application
CMD ["python", "app.py"]
