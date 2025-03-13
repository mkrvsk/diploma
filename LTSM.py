from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from data_visualization_and_processing import X_train_mfcc, X_test_mfcc, CLASSES, y_train, y_test

X_train_lstm = X_train_mfcc.reshape((X_train_mfcc.shape[0], X_train_mfcc.shape[1], 1))
X_test_lstm = X_test_mfcc.reshape((X_test_mfcc.shape[0], X_test_mfcc.shape[1], 1))

lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
    LSTM(64),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(len(CLASSES), activation='softmax')
])

lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=16, validation_data=(X_test_lstm, y_test))

loss, acc = lstm_model.evaluate(X_test_lstm, y_test)
print(f"LSTM Accuracy: {acc}")