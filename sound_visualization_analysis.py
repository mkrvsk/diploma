import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# Шлях до папки з аудіофайлами
data_path = "/Users/Mkrvsk/Desktop/diploma/sounds/ambulance"

audio_files = [f for f in os.listdir(data_path) if f.endswith('.wav')]

file_path = os.path.join(data_path, audio_files[0])
y, sr = librosa.load(file_path, sr=None)

# 1. Візуалізація спектрограми
D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
plt.figure(figsize=(12, 4))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title("Спектрограма (STFT)")
plt.xlabel("Час (сек)")
plt.ylabel("Частота (Hz)")
plt.show()

# 2. Візуалізація мел-спектрограми
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
plt.figure(figsize=(12, 4))
librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Мел-спектрограма")
plt.xlabel("Час (сек)")
plt.ylabel("Мел-шкала")
plt.show()

# 3. Візуалізація MFCC
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
plt.figure(figsize=(12, 4))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.colorbar()
plt.title("MFCC (Mel-Frequency Cepstral Coefficients)")
plt.xlabel("Час (сек)")
plt.ylabel("MFCC коефіцієнти")
plt.show()

# 4. Візуалізація звукової хвилі (Waveform)
plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Звукова хвиля")
plt.xlabel("Час (сек)")
plt.ylabel("Амплітуда")
plt.show()

# 5. Візуалізація частотного спектра (Frequency Spectrum)
D = np.abs(librosa.stft(y))
plt.figure(figsize=(12, 4))
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title("Частотний спектр")
plt.xlabel("Час (сек)")
plt.ylabel("Частота (Hz)")
plt.show()

# 6. Візуалізація хрома-функції (Chromagram)
chromagram = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
plt.figure(figsize=(12, 4))
librosa.display.specshow(chromagram, sr=sr, x_axis='time', y_axis='chroma')
plt.colorbar()
plt.title("Хрома-функція")
plt.xlabel("Час (сек)")
plt.ylabel("Тонові групи")
plt.show()

# 7. Візуалізація тонального спектра (Tonnetz)
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
plt.figure(figsize=(12, 4))
librosa.display.specshow(tonnetz, x_axis='time')
plt.colorbar()
plt.title("Тональний спектр (Tonnetz)")
plt.xlabel("Час (сек)")
plt.ylabel("Тональні компоненти")
plt.show()
