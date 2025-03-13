import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "/Users/Mkrvsk/Desktop/diploma/sounds"
CLASSES = ["ambulance", "firetruck", "traffic"]

def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)  # Усереднення по часу

def extract_mel_spectrogram(file_path, n_mels=128):
    y, sr = librosa.load(file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return np.mean(mel_spec, axis=1)  # Усереднення по часу

def extract_spectrogram(file_path, n_mels=128, fixed_frames=500):
    y, sr = librosa.load(file_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)

    if S_db.shape[1] > fixed_frames:  # Обрізати
        S_db = S_db[:, :fixed_frames]
    elif S_db.shape[1] < fixed_frames:  # Доповнити нулями
        pad_width = fixed_frames - S_db.shape[1]
        S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant')

    return S_db

X_mfcc, X_mel, X_spectrogram, y_labels = [], [], [], []

for idx, class_name in enumerate(CLASSES):
    class_path = os.path.join(DATA_PATH, class_name)
    for file_name in os.listdir(class_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(class_path, file_name)

            # Отримати MFCC, MEL та спектрограму
            X_mfcc.append(extract_mfcc(file_path))
            X_mel.append(extract_mel_spectrogram(file_path))
            X_spectrogram.append(extract_spectrogram(file_path))

            # Додати мітку класу
            y_labels.append(idx)

X_mfcc = np.array(X_mfcc)
X_mel = np.array(X_mel)
X_spectrogram = np.array(X_spectrogram)
y_labels = np.array(y_labels)

scaler = StandardScaler()
X_mfcc = scaler.fit_transform(X_mfcc)
X_mel = scaler.fit_transform(X_mel)

X_train_mfcc, X_test_mfcc, y_train, y_test = train_test_split(X_mfcc, y_labels, test_size=0.2, random_state=42)
X_train_mel, X_test_mel, _, _ = train_test_split(X_mel, y_labels, test_size=0.2, random_state=42)
X_train_spectrogram, X_test_spectrogram, _, _ = train_test_split(X_spectrogram, y_labels, test_size=0.2, random_state=42)

print("Data is ready!")