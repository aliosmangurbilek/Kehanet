# model doğruluk oranını hesaplar
from core.tensor import Tensor
from data.dataloader import load_mnist_local, DataLoader
import numpy as np

# 1) Test setini yükle (one_hot=False, tekil etiket olarak)
_, test_ds = load_mnist_local('/home/ali/PycharmProjects/Kehanet/datasets/mnist', normalize=True, one_hot=False)
X_test, y_test = test_ds.data, test_ds.labels   # (10000,784), (10000,)

# 2) Modeli yükleyin / tanımlayın
import pickle
with open("/home/ali/PycharmProjects/Kehanet/examples/trained_mnist_model.pkl", "rb") as f:
    model = pickle.load(f)

# 3) Toplu tahmin / doğruluk
X = Tensor(X_test, requires_grad=False)
logits = model(X).data            # (10000,10)
preds  = np.argmax(logits, axis=1)
acc    = np.mean(preds == y_test)
print(f"Test set accuracy: {acc*100:.2f}%")
