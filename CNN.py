import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from data_visualization_and_processing import y_train, X_train_spectrogram, X_test_spectrogram, y_test, CLASSES

X_train_spectrogram = X_train_spectrogram[..., np.newaxis]  # Додаємо канал
X_test_spectrogram = X_test_spectrogram[..., np.newaxis]
y_train_cnn = to_categorical(y_train, num_classes=len(CLASSES))
y_test_cnn = to_categorical(y_test, num_classes=len(CLASSES))

cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X_train_spectrogram.shape[1], X_train_spectrogram.shape[2], 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(CLASSES), activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cnn_model.fit(X_train_spectrogram, y_train_cnn, epochs=10, batch_size=16,
              validation_data=(X_test_spectrogram, y_test_cnn))

loss, acc = cnn_model.evaluate(X_test_spectrogram, y_test_cnn)
print(f"CNN Accuracy: {acc}")
