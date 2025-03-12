import librosa
import numpy as np
import os
import random
import soundfile as sf

# 1. Додавання шуму
def add_noise(y, noise_level=0.005):
    noise = np.random.randn(len(y))
    y_noisy = y + noise_level * noise
    return np.clip(y_noisy, -1.0, 1.0)

# 2. Зміна швидкості
def change_speed(y, rate=1.5):
    return librosa.effects.time_stretch(y, rate=1.5)

# 3. Зміна висоти
def change_pitch(y, sr, pitch_factor=2):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_factor)


# 4. Часова трансформація
def time_shift(y, shift_max=2):
    shift = random.randint(-shift_max, shift_max)  # випадковий зсув в межах shift_max
    return np.roll(y, shift)

def augment_audio_files(data_path, class_names, output_path, augment_factor=1):
    for class_name in class_names:
        class_folder = os.path.join(data_path, class_name)
        audio_files = [f for f in os.listdir(class_folder) if f.endswith('.wav')]

        for audio_file in audio_files:
            file_path = os.path.join(class_folder, audio_file)
            y, sr = librosa.load(file_path, sr=None)

            # Створимо кілька аугментованих варіантів
            for i in range(augment_factor):
                # 1. Додавання шуму
                y_noisy = add_noise(y)

                # 2. Зміна швидкості
                y_speed = change_speed(y)

                # 3. Зміна висоти
                y_pitch = change_pitch(y, sr)

                # 4. Часова трансформація
                y_shifted = time_shift(y)

                # Збережемо аугментовані файли
                base_file_name = audio_file.replace('.wav', '')

                # Збереження аугментованих аудіофайлів
                sf.write(os.path.join(output_path, class_name, f"{base_file_name}_noisy_{i}.wav"), y_noisy, sr)
                sf.write(os.path.join(output_path, class_name, f"{base_file_name}_speed_{i}.wav"), y_speed, sr)
                sf.write(os.path.join(output_path, class_name, f"{base_file_name}_pitch_{i}.wav"), y_pitch, sr)
                sf.write(os.path.join(output_path, class_name, f"{base_file_name}_shifted_{i}.wav"), y_shifted, sr)

                print(f"Processed and saved {base_file_name} with augmentations.")

data_path = "/Users/Mkrvsk/Desktop/diploma/sounds"
output_path = "/Users/Mkrvsk/Desktop/diploma/sounds_augmented"  # Шлях до папки для збереження аугментованих даних
class_names = ['Ambulance', 'Firetruck', 'Traffic']

for class_name in class_names:
    os.makedirs(os.path.join(output_path, class_name), exist_ok=True)

# Запуск аугментації аудіофайлів
augment_audio_files(data_path, class_names, output_path, augment_factor=2)