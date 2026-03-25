from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def run_unsupervised_model(X, y):
    # Train K-Means (no labels used here)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)

    clusters = kmeans.labels_

    # Map clusters to labels (for evaluation only)
    predicted_labels = np.zeros_like(clusters)

    for i in range(2):
        mask = (clusters == i)
        if np.sum(y[mask]) > len(y[mask]) / 2:
            predicted_labels[mask] = 1
        else:
            predicted_labels[mask] = 0

    # Results
    accuracy = accuracy_score(y, predicted_labels)
    report = classification_report(y, predicted_labels)

    print("\n=== K-Means (Unsupervised) ===")
    print("Accuracy:", accuracy)
    print(report)

    return accuracy