import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "/Users/Mkrvsk/Desktop/diploma/sounds"
CLASSES = ["ambulance", "firetruck", "traffic"]


def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)  # Average over time


# Function to extract and normalize spectrograms, with padding/trimming
def extract_spectrogram(file_path, n_mels=128, fixed_frames=500):
    y, sr = librosa.load(file_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Enforce a fixed shape for all spectrograms (padding or trimming)
    if S_db.shape[1] > fixed_frames:  # Trim to fixed_frames
        S_db = S_db[:, :fixed_frames]
    elif S_db.shape[1] < fixed_frames:  # Pad with zeros to fixed_frames
        pad_width = fixed_frames - S_db.shape[1]
        S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant')

    return S_db


X_mfcc, X_spectrogram, y_labels = [], [], []
for idx, class_name in enumerate(CLASSES):
    class_path = os.path.join(DATA_PATH, class_name)
    for file_name in os.listdir(class_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(class_path, file_name)

            # Get MFCC features
            mfcc_features = extract_mfcc(file_path)
            X_mfcc.append(mfcc_features)

            # Get spectrograms
            spectrogram = extract_spectrogram(file_path)
            X_spectrogram.append(spectrogram)

            # Add label
            y_labels.append(idx)

# Convert lists to NumPy arrays
X_mfcc = np.array(X_mfcc)
X_spectrogram = np.array(X_spectrogram)
y_labels = np.array(y_labels)

# Scale MFCCs for use in SVM/LSTM
scaler = StandardScaler()
X_mfcc = scaler.fit_transform(X_mfcc)

X_train_mfcc, X_test_mfcc, y_train, y_test = train_test_split(X_mfcc, y_labels, test_size=0.2, random_state=42)
X_train_spectrogram, X_test_spectrogram, _, _ = train_test_split(X_spectrogram, y_labels, test_size=0.2,
                                                                 random_state=42)

print("Data is ready!")
