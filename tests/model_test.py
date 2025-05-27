# model doğruluk oranını hesaplar
from core.tensor import Tensor
from data.dataloader import load_mnist_local, DataLoader
import numpy as np
import os

# Ensure necessary directories exist
os.makedirs('/home/ali/PycharmProjects/Kehanet/outputs', exist_ok=True)

# Import sklearn metrics
try:
    from sklearn.metrics import (
        classification_report, confusion_matrix, f1_score, 
        precision_score, recall_score, silhouette_score,
        roc_curve, auc
    )
    from sklearn.preprocessing import label_binarize
    has_sklearn = True
except ImportError:
    print("sklearn not found. Limited metrics will be available.")
    has_sklearn = False

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    has_viz = True
except ImportError:
    print("Visualization libraries not found. Plots will not be generated.")
    has_viz = False

# 1) Test setini yükle (one_hot=False, tekil etiket olarak)
_, test_ds = load_mnist_local('/home/ali/PycharmProjects/Kehanet/datasets/mnist', normalize=True, one_hot=False)
X_test, y_test = test_ds.data, test_ds.labels   # (10000,784), (10000,)

# 2) Modeli yükleyin / tanımlayın
import pickle
with open("/home/ali/PycharmProjects/Kehanet/models/trained_mnist_model.pkl", "rb") as f:
    model = pickle.load(f)

# 3) Toplu tahmin / doğruluk
X = Tensor(X_test, requires_grad=False)
logits = model(X).data            # (10000,10)
preds = np.argmax(logits, axis=1)
acc = np.mean(preds == y_test)
print(f"Test set accuracy: {acc*100:.2f}%")

# 4) Detailed metrics (if sklearn is available)
if has_sklearn:
    # Calculate F1 scores
    f1_macro = f1_score(y_test, preds, average='macro')
    f1_weighted = f1_score(y_test, preds, average='weighted')
    print(f"Macro F1 Score: {f1_macro:.4f}")
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    
    # Precision and Recall
    precision = precision_score(y_test, preds, average='macro')
    recall = recall_score(y_test, preds, average='macro')
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Per-class F1 scores
    f1_per_class = f1_score(y_test, preds, average=None)
    print("\nF1 Score per class:")
    for i, score in enumerate(f1_per_class):
        print(f"Digit {i}: {score:.4f}")
    
    # Confusion Matrix as text
    cm = confusion_matrix(y_test, preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification Report
    report = classification_report(y_test, preds)
    print("\nClassification Report:")
    print(report)
    
    # Try calculating silhouette score
    try:
        # Using softmax probabilities instead of raw logits
        softmax_probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        silhouette_avg = silhouette_score(softmax_probs, preds)
        print(f"\nSilhouette Score: {silhouette_avg:.4f}")
        print("Note: Silhouette score is typically used for clustering evaluation,")
        print("      so interpret with caution for classification tasks.")
    except Exception as e:
        print(f"\nCouldn't calculate Silhouette Score: {e}")

# 5) Visualizations (if visualization libraries are available)
if has_sklearn and has_viz:
    # Confusion Matrix visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('/home/ali/PycharmProjects/Kehanet/outputs/confusion_matrix.png')
    print(f"\nConfusion matrix visualization saved to /home/ali/PycharmProjects/Kehanet/outputs/confusion_matrix.png")
    
    # Class distribution analysis
    class_counts = np.bincount(y_test)
    class_accuracy = np.zeros(10)
    for i in range(10):
        class_indices = np.where(y_test == i)[0]
        class_accuracy[i] = np.mean(preds[class_indices] == y_test[class_indices])
    
    # Plot class distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(10), class_counts)
    plt.title('Class Distribution in Test Set')
    plt.xlabel('Digit')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(10), class_accuracy * 100)
    plt.title('Accuracy per Class')
    plt.xlabel('Digit')
    plt.ylabel('Accuracy (%)')
    plt.ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('/home/ali/PycharmProjects/Kehanet/outputs/class_analysis.png')
    print(f"\nClass analysis saved to /home/ali/PycharmProjects/Kehanet/outputs/class_analysis.png")
    
    # Error Analysis - Examples of misclassified digits
    misclassified_indices = np.where(preds != y_test)[0]
    if len(misclassified_indices) > 0:
        print(f"\nNumber of misclassified examples: {len(misclassified_indices)}")
        
        # Sample up to 10 misclassified examples
        sample_size = min(10, len(misclassified_indices))
        sample_indices = np.random.choice(misclassified_indices, sample_size, replace=False)
        
        plt.figure(figsize=(15, 8))
        for i, idx in enumerate(sample_indices):
            plt.subplot(2, 5, i+1)
            # Reshape back to 28x28 for visualization
            img = X_test[idx].reshape(28, 28)
            plt.imshow(img, cmap='gray')
            plt.title(f"True: {y_test[idx]}, Pred: {preds[idx]}")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('/home/ali/PycharmProjects/Kehanet/outputs/misclassified_examples.png')
        print(f"Misclassified examples saved to /home/ali/PycharmProjects/Kehanet/outputs/misclassified_examples.png")
else:
    print("\nInstall sklearn, matplotlib, and seaborn for more detailed analysis and visualizations.")
