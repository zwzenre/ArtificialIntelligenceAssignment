import tkinter as tk
from tkinter import ttk

def run_gui(vectorizer, nb_model, svm_model, km_model, cluster_to_label):

    def show_result(name, prediction):
        row = tk.Frame(result_frame)
        row.pack(anchor="w")

        tk.Label(row, text=f"{name:<12}:", width=15, anchor="w").pack(side="left")

        if prediction == 1:
            tk.Label(row, text="Spam", fg="red").pack(side="left")
        elif prediction == 0:
            tk.Label(row, text="Ham", fg="green").pack(side="left")
        else:
            tk.Label(row, text=prediction, fg="gray").pack(side="left")

    def predict_email():
        email_text = text_input.get("1.0", tk.END)
        email_vector = vectorizer.transform([email_text])

        for widget in result_frame.winfo_children():
            widget.destroy()

        algo = algo_var.get()

        # ===== Naive Bayes =====
        if algo == "Naive Bayes":
            nb_pred = nb_model.predict(email_vector)[0]
            show_result("Naive Bayes", nb_pred)

        # ===== SVM =====
        elif algo == "SVM":
            svm_pred = svm_model.predict(email_vector)[0]
            show_result("SVM", svm_pred)

        # ===== K-Means =====
        elif algo == "K-Means":
            cluster = km_model.predict(email_vector)[0]
            km_pred = cluster_to_label[cluster]
            show_result("K-Means", km_pred)

        # ===== DBSCAN =====
        elif algo == "DBSCAN":
            show_result("DBSCAN", "N/A")

        # ===== Run All =====
        elif algo == "Run All":
            nb_pred = nb_model.predict(email_vector)[0]
            show_result("Naive Bayes", nb_pred)

            svm_pred = svm_model.predict(email_vector)[0]
            show_result("SVM", svm_pred)

            cluster = km_model.predict(email_vector)[0]
            km_pred = cluster_to_label[cluster]
            show_result("K-Means", km_pred)

            show_result("DBSCAN", "N/A")

    # ===== GUI =====
    root = tk.Tk()
    root.title("Spam Email Detection")

    tk.Label(root, text="Enter Email:").pack(anchor="w")

    text_input = tk.Text(root, height=10, width=50)
    text_input.pack()

    tk.Label(root, text="Select Algorithm:").pack(anchor="w")

    algo_var = tk.StringVar()
    algo_menu = ttk.Combobox(root, textvariable=algo_var, state="readonly")
    algo_menu['values'] = ("Naive Bayes", "SVM", "K-Means", "DBSCAN", "Run All")
    algo_menu.current(0)
    algo_menu.pack()

    tk.Button(root, text="Predict", command=predict_email).pack(pady=5)

    result_frame = tk.Frame(root)
    result_frame.pack(anchor="w", fill="x", padx=10, pady=5)

    root.mainloop()