from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Naive Bayes Model (NG ZHE WEI)
def run_naive_bayes(x_train, x_test, y_train, y_test):
    # Train model
    model = MultinomialNB()
    model.fit(x_train, y_train)

    # Predict
    y_pred = model.predict(x_test)

    # Results
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("=== Naive Bayes (Supervised) ===")
    print("Accuracy:", accuracy)
    print(report)

    return model, accuracy

# SVM Model (KHOO KAH QIN)
def run_svm(x_train, x_test, y_train, y_test):

    model = SVC(probability=True)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("=== SVM (Supervised) ===")
    print("Accuracy:", accuracy)
    print(report)

    return model, accuracy