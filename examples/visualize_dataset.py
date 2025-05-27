import numpy as np
import matplotlib.pyplot as plt

# Eğer kendi load_mnist_local fonksiyonunuzu kullanacaksanız:
from data.dataloader import load_mnist_local

# 1. Veri setini yükle (one_hot=False, raw etiketler olsun)
train_ds, test_ds = load_mnist_local('/home/ali/PycharmProjects/Kehanet/datasets/mnist', normalize=True, one_hot=False)
X_train, y_train = train_ds.data, train_ds.labels
X_test,  y_test  = test_ds.data,  test_ds.labels

# 2. Boyutlara bakalım
print("X_train.shape:", X_train.shape)   # (60000, 784)
print("y_train.shape:", y_train.shape)   # (60000,)
print("X_test.shape: ", X_test.shape)    # (10000, 784)
print("y_test.shape: ", y_test.shape)    # (10000,)

# 3. Etiket dağılımı
counts = np.bincount(y_train.astype(int))
for digit, cnt in enumerate(counts):
    print(f"Label {digit}: {cnt} örnek")

# 4. Görselleri 28×28’e çevir ve ilk örnekleri göster
X_train_img = X_train.reshape(-1, 28, 28)

num_classes = 10
fig, axes = plt.subplots(1, num_classes, figsize=(20, 3))
for i in range(num_classes):
    # i sınıfından ilk örneği al
    sample = X_train_img[y_train.astype(int) == i][0]
    axes[i].imshow(sample, cmap='gray')
    axes[i].set_title(f"Label: {i}", fontsize=12)
    axes[i].axis('off')
plt.show()
