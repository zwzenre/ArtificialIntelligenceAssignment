import tkinter as tk
from tkinter import ttk

# (HENG WEI YI)
def run_gui(vectorizer, nb_model, svm_model, km_model, cluster_to_label,
            nb_acc, svm_acc, km_acc,
            db_acc, db_clusters, db_noise):

    def show_result(name, prediction, confidence=None, accuracy=None):
        row = tk.Frame(result_frame)
        row.pack(anchor="w")

        tk.Label(row, text=f"{name:<12}:", width=15, anchor="w").pack(side="left")

        # Label
        if prediction == 1:
            label_text = "Spam"
            color = "red"
        elif prediction == 0:
            label_text = "Ham"
            color = "green"
        else:
            label_text = str(prediction)
            color = "gray"

        # Confidence
        if confidence is not None:
            label_text += f" ({confidence:.2f}%)"

        # Accuracy
        if accuracy is not None:
            label_text += f" | Accuracy: {accuracy:.2f}%"

        tk.Label(row, text=label_text, fg=color).pack(side="left")

    def predict_email():
        email_text = text_input.get("1.0", tk.END)
        email_vector = vectorizer.transform([email_text])

        # Clear previous results
        for widget in result_frame.winfo_children():
            widget.destroy()

        algo = algo_var.get()

        # ===== Naive Bayes =====
        if algo == "Naive Bayes":
            nb_pred = nb_model.predict(email_vector)[0]
            nb_conf = nb_model.predict_proba(email_vector)[0].max() * 100
            show_result("Naive Bayes", nb_pred, nb_conf, nb_acc)

        # ===== SVM =====
        elif algo == "SVM":
            svm_pred = svm_model.predict(email_vector)[0]
            svm_conf = svm_model.predict_proba(email_vector)[0].max() * 100
            show_result("SVM", svm_pred, svm_conf, svm_acc)

        # ===== K-Means =====
        elif algo == "K-Means":
            cluster = km_model.predict(email_vector)[0]
            km_pred = cluster_to_label[cluster]

            distance = km_model.transform(email_vector)[0][cluster]
            km_conf = (1 / (1 + distance)) * 100

            show_result("K-Means", km_pred, km_conf, km_acc)

        # ===== DBSCAN =====
        elif algo == "DBSCAN":
            db_text = f"Clusters: {db_clusters}, Noise: {db_noise}"
            show_result("DBSCAN", db_text, None, db_acc)

        # ===== Run All =====
        elif algo == "Run All":

            # Naive Bayes
            nb_pred = nb_model.predict(email_vector)[0]
            nb_conf = nb_model.predict_proba(email_vector)[0].max() * 100
            show_result("Naive Bayes", nb_pred, nb_conf, nb_acc)

            # SVM
            svm_pred = svm_model.predict(email_vector)[0]
            svm_conf = svm_model.predict_proba(email_vector)[0].max() * 100
            show_result("SVM", svm_pred, svm_conf, svm_acc)

            # K-Means
            cluster = km_model.predict(email_vector)[0]
            km_pred = cluster_to_label[cluster]
            distance = km_model.transform(email_vector)[0][cluster]
            km_conf = (1 / (1 + distance)) * 100
            show_result("K-Means", km_pred, km_conf, km_acc)

            # DBSCAN
            db_text = f"Clusters: {db_clusters}, Noise: {db_noise}"
            show_result("DBSCAN", db_text, None, db_acc)

    # ===== GUI =====
    root = tk.Tk()
    root.title("Spam Email Detection")
    root.geometry("900x600")

    main_frame = tk.Frame(root)
    main_frame.pack(expand=True)

    tk.Label(main_frame, text="Enter Email:").pack(pady=5)

    text_input = tk.Text(main_frame, height=15, width=80)
    text_input.pack(pady=10)

    tk.Label(main_frame, text="Select Algorithm:").pack(pady=5)

    algo_var = tk.StringVar()
    algo_menu = ttk.Combobox(main_frame, textvariable=algo_var, state="readonly")
    algo_menu['values'] = ("Naive Bayes", "SVM", "K-Means", "DBSCAN", "Run All")
    algo_menu.current(0)
    algo_menu.pack(pady=5)

    tk.Button(main_frame, text="Predict", command=predict_email).pack(pady=10)

    result_frame = tk.Frame(main_frame)
    result_frame.pack(pady=10)

    root.mainloop()