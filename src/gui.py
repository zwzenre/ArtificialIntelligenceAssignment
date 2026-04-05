import tkinter as tk
from tkinter import ttk

def run_gui(vectorizer, nb_model, svm_model):
    def predict_email():
        email_text = text_input.get("1.0", tk.END)

        # 👉 转 TF-IDF
        email_vector = vectorizer.transform([email_text])

        algo = algo_var.get()

        if algo == "Naive Bayes":
            prediction = nb_model.predict(email_vector)[0]

        elif algo == "SVM":
            prediction = svm_model.predict(email_vector)[0]

        elif algo == "K-Means":
            prediction = "Clustered (Unsupervised)"

        elif algo == "DBSCAN":
            prediction = "Clustered (Unsupervised)"

        # 👉 转回 label
        if prediction == 1:
            result = "Spam"
        elif prediction == 0:
            result = "Ham"
        else:
            result = prediction

        result_label.config(text=f"Prediction: {result}")

    # ===== GUI =====
    root = tk.Tk()
    root.title("Spam Email Detection")

    tk.Label(root, text="Enter Email:").pack()

    text_input = tk.Text(root, height=10, width=50)
    text_input.pack()

    tk.Label(root, text="Select Algorithm:").pack()

    algo_var = tk.StringVar()
    algo_menu = ttk.Combobox(root, textvariable=algo_var)
    algo_menu['values'] = ("Naive Bayes", "SVM", "K-Means", "DBSCAN")
    algo_menu.current(0)
    algo_menu.pack()

    tk.Button(root, text="Predict", command=predict_email).pack()

    result_label = tk.Label(root, text="Prediction: ")
    result_label.pack()

    root.mainloop()