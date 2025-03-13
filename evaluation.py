import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
from CNN import cnn_model

def evaluate_model(y_true, y_pred, cnn_model):
    """
    Функція для оцінки моделі: точність, precision, recall, F1-score і матриця помилок.

    :param y_true: Справжні мітки класів
    :param y_pred: Передбачені мітки класів
    :param model_name: Назва моделі (str)
    """
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"📊 {cnn_model} Performance Metrics:")
    print(f"🔹 Accuracy: {acc:.4f}")
    print(f"🔹 Precision: {precision:.4f}")
    print(f"🔹 Recall: {recall:.4f}")
    print(f"🔹 F1-score: {f1:.4f}\n")

    print(f"{cnn_model} Classification Report:\n")
    print(classification_report(y_true, y_pred))

    # Побудова матриці помилок
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Ambulance", "Firetruck", "Traffic"],
                yticklabels=["Ambulance", "Firetruck", "Traffic"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{cnn_model} Confusion Matrix")
    plt.show()