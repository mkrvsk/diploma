from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from data_visualization_and_processing import y_train, X_train_mfcc, X_test_mfcc, y_test

svm_model = SVC(kernel="linear")
svm_model.fit(X_train_mfcc, y_train)

y_pred_svm = svm_model.predict(X_test_mfcc)

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

from evaluation import evaluate_model

evaluate_model(y_test, y_pred_svm, "SVM")