import numpy as np
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, LSTM, Reshape
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report

from data_preparation import X_train_mfcc, X_test_mfcc, X_train_mel, X_test_mel, X_train_spectrogram, X_test_spectrogram, y_train, y_test

num_classes = len(np.unique(y_train))
y_train_categorical = to_categorical(y_train, num_classes)
y_test_categorical = to_categorical(y_test, num_classes)

# 1. SVM на MFCC
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_mfcc, y_train)
svm_preds = svm_model.predict(X_test_mfcc)
print("SVM Accuracy:", accuracy_score(y_test, svm_preds))
print("SVM Classification Report:\n", classification_report(y_test, svm_preds))

# 2. LSTM на MEL-спектрограмі
lstm_model = Sequential([
    Reshape((X_train_mel.shape[1], 1), input_shape=(X_train_mel.shape[1],)),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(num_classes, activation='softmax')
])
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_mel, y_train_categorical, epochs=10, batch_size=32, validation_split=0.2)
lstm_accuracy = lstm_model.evaluate(X_test_mel, y_test_categorical)[1]
print("LSTM Accuracy:", lstm_accuracy)

# 3. CNN на спектрограмах
X_train_spectrogram = X_train_spectrogram[..., np.newaxis]  # Додаємо канал для CNN
X_test_spectrogram = X_test_spectrogram[..., np.newaxis]

cnn_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X_train_spectrogram.shape[1:]),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_spectrogram, y_train_categorical, epochs=10, batch_size=32, validation_split=0.2)
cnn_accuracy = cnn_model.evaluate(X_test_spectrogram, y_test_categorical)[1]
print("CNN Accuracy:", cnn_accuracy)

# Визначення найкращого методу
results = {
    "SVM (MFCC)": accuracy_score(y_test, svm_preds),
    "LSTM (MEL)": lstm_accuracy,
    "CNN (Spectrogram)": cnn_accuracy
}
best_model = max(results, key=results.get)
print("Best Model:", best_model)
