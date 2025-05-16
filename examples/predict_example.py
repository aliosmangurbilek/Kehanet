# examples/predict_example.py
"""
Kamera veya dosyadan bir el yazısı rakam resmi alıp,
Kehanet kütüphanesi ile tahmin yapan örnek script.
"""
import cv2
import numpy as np
from PIL import Image
from core.tensor import Tensor
from layers.dense import Dense
from core.autograd import relu
import pickle  # Import pickle to load the trained model
from core.sequential import SimpleSequential  # Updated import

# --- Model tanımı: MNIST eğitilmiş ağı yükleyin ---
def load_trained_model(model_path: str):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model_path = '/home/ali/PycharmProjects/Kehanet/models/trained_mnist_model.pkl'
try:
    model = load_trained_model(model_path)
    print(f"Trained model loaded from {model_path}")
except FileNotFoundError:
    print(f"Trained model not found at {model_path}. Using an untrained model.")

# --- Ön işleme fonksiyonu ---

def preprocess(img: np.ndarray) -> np.ndarray:
    # Gri ton -> 28x28
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ROI olarak küçük pencereyi merkeze al
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    # Normalize 0-1, koyu-zemin inversion gerekebilir
    norm = resized.astype(np.float32) / 255.0
    # Flatten
    return norm.reshape(1, 28*28)

# --- Kamera üzerinden canlı tahmin ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kameraya erişilemiyor, resim dosyası kullanın.")
else:
    print("Çıkmak için 'q' tuşuna basın.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Tahmin bölgesi: ekranın ortasında 200x200 dikdörtgen
        h, w = frame.shape[:2]
        x0, y0 = w//2-100, h//2-100
        roi = frame[y0:y0+200, x0:x0+200]  # Only this 200x200 region is used for detection
        prep = preprocess(roi)  # Preprocessing only the ROI
        x_tensor = Tensor(prep, requires_grad=False)
        preds = model(x_tensor).data  # Model prediction only on the ROI
        digit = np.argmax(preds, axis=1)[0]
        # Görüntüye çizim
        cv2.rectangle(frame, (x0,y0), (x0+200,y0+200), (0,255,0), 2)  # This draws the green square
        cv2.putText(frame, f"Pred: {digit}", (x0, y0-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Kehanet Digit Predictor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# --- Dosyadan resim ile tahmin (opsiyonel) ---
def predict_from_file(path: str):
    img = cv2.imread(path)
    prep = preprocess(img)
    x = Tensor(prep, requires_grad=False)
    preds = model(x).data
    print(f"Image '{path}' prediction: {np.argmax(preds, axis=1)[0]}")

# Örnek kullanım:
# predict_from_file('digit.png')

