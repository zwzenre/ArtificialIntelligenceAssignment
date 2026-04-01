import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.supervised_model import run_naive_bayes, run_svm
from src.unsupervised_model import run_kmeans, run_dbscan

# Load dataset
data = pd.read_csv("data/spam.csv")

# Fix column names if needed
data.columns = ['label', 'message']

# Remove rows with missing messages
data = data.dropna(subset=['message'])

# Convert labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})


# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])
y = data['label']

# Run models
nb_accuracy = run_naive_bayes(X, y)
svm_accuracy = run_svm(X, y)

km_accuracy = run_kmeans(X, y)
db_accuracy = run_dbscan(X, y)

# Compare
print("\n=== Comparison ===")
print("Naive Bayes Accuracy:", nb_accuracy)
print("SVM Accuracy:", svm_accuracy)

print("K-Means Accuracy:", km_accuracy)
print("DBSCAN Accuracy:", db_accuracy)