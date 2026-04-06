from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# K-Means Model
def run_kmeans(x_train, x_test, y_train, y_test):
    # Train K-Means (no labels used here)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(x_train)

    clusters = kmeans.predict(x_test)

    # Map clusters to labels (for evaluation only)
    predicted_labels = np.zeros_like(clusters)

    for i in range(2):
        mask = (clusters == i)
        if np.sum(y_test[mask]) > len(y_test[mask]) / 2:
            predicted_labels[mask] = 1
        else:
            predicted_labels[mask] = 0

    # use training data to do mapping
    train_clusters = kmeans.predict(x_train)

    cluster_to_label = {}

    for c in np.unique(train_clusters):
        mask = (train_clusters == c)
        cluster_to_label[c] = np.bincount(y_train[mask]).argmax()

    # Results
    accuracy = accuracy_score(y_test, predicted_labels)
    report = classification_report(y_test, predicted_labels)

    print("\n=== K-Means (Unsupervised) ===")
    print("Accuracy:", accuracy)
    print(report)

    return kmeans, cluster_to_label, accuracy

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