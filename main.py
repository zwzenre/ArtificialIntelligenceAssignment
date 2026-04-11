import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.supervised_model import run_naive_bayes, run_svm
from src.unsupervised_model import run_kmeans, run_dbscan
from src.gui import run_gui

# Load dataset
data = pd.read_csv("data/spam.csv")

# Fix column names
data.columns = ['label', 'message']

# Remove missing values
data = data.dropna(subset=['message'])

# Convert labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
x = vectorizer.fit_transform(data['message'])
y = data['label']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1
)

# ===== Train Models =====
nb_model, nb_accuracy = run_naive_bayes(x_train, x_test, y_train, y_test)
svm_model, svm_accuracy = run_svm(x_train, x_test, y_train, y_test)

km_model, cluster_to_label, km_accuracy = run_kmeans(
    x_train, x_test, y_train, y_test
)

db_accuracy, db_clusters, db_noise = run_dbscan(x, y)

# ===== Print Comparison =====
print("\n=== Comparison ===")
print("Naive Bayes Accuracy:", nb_accuracy)
print("SVM Accuracy:", svm_accuracy)
print("K-Means Accuracy:", km_accuracy)
print("DBSCAN Accuracy:", db_accuracy)
print("Clusters:", db_clusters)
print("Noise Points:", db_noise)

# ===== Run GUI =====
run_gui(
    vectorizer,
    nb_model,
    svm_model,
    km_model,
    cluster_to_label,
    nb_accuracy * 100,
    svm_accuracy * 100,
    km_accuracy * 100,
    db_accuracy * 100,
    db_clusters,
    db_noise
)