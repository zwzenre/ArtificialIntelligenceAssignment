from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# K-Means Model
def run_kmeans(X_train, X_test, y_train, y_test):
    # Train K-Means (no labels used here)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_train)

    clusters = kmeans.predict(X_test)

    # Map clusters to labels (for evaluation only)
    predicted_labels = np.zeros_like(clusters)

    for i in range(2):
        mask = (clusters == i)
        if np.sum(y_test[mask]) > len(y_test[mask]) / 2:
            predicted_labels[mask] = 1
        else:
            predicted_labels[mask] = 0

    # Results
    accuracy = accuracy_score(y_test, predicted_labels)
    report = classification_report(y_test, predicted_labels)

    print("\n=== K-Means (Unsupervised) ===")
    print("Accuracy:", accuracy)
    print(report)

    return accuracy

# DBSCAN Model
def run_dbscan(x,y):
    model = DBSCAN(eps=0.5, min_samples=5)
    clusters = model.fit_predict(x)

    # Replace noise (-1) with a label (e.g., spam = 1)
    clusters = np.where(clusters == -1, 1, clusters)

    # Map clusters to actual labels
    mapped_labels = np.zeros_like(clusters)

    for cluster in np.unique(clusters):
        mask = (clusters == cluster)
        mapped_labels[mask] = np.bincount(y[mask]).argmax()

    accuracy = accuracy_score(y, mapped_labels)
    report = classification_report(y, mapped_labels)

    print("\n=== DBSCAN (Unsupervised) ===")
    print("Accuracy:", accuracy)
    print(report)

    return accuracy