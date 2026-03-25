from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def run_supervised_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Results
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("=== Naive Bayes (Supervised) ===")
    print("Accuracy:", accuracy)
    print(report)

    return accuracy