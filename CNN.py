import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models

DATASET_PATH = r'D:\GERALD ILYAS\SEMESTER 4\AI\genres'  
SAMPLE_RATE = 22050
DURATION = 30  
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def extract_features(dataset_path, n_mfcc=13):
    features = []
    labels = []
    filenames = []
    
    for genre in os.listdir(dataset_path):
        genre_path = os.path.join(dataset_path, genre)
        if not os.path.isdir(genre_path):
            continue
        for file in os.listdir(genre_path):
            if not file.endswith('.au'):
                continue
            file_path = os.path.join(genre_path, file)
            try:
                y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                mfcc_mean = np.mean(mfcc.T, axis=0)
                features.append(mfcc_mean)
                labels.append(genre)
                filenames.append(file)
            except Exception as e:
                print(f"⚠️ Gagal membaca {file}: {e}")
    
    return np.array(features), np.array(labels), filenames

X, y, filenames = extract_features(DATASET_PATH)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
genre_list = encoder.classes_

X_train, X_test, y_train, y_test, file_train, file_test = train_test_split(
    X, y_encoded, filenames, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(genre_list), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2, verbose=0)

y_pred = np.argmax(model.predict(X_test), axis=1)

y_true_genres = [genre_list[i] for i in y_test]
y_pred_genres = [genre_list[i] for i in y_pred]

print("\n🎧 Hasil Klasifikasi Genre Musik")
print(f"{'Nama File':<25} {'Genre Asli':<15} {'Prediksi':<15}")
print("-" * 55)
for file, asli, prediksi in zip(file_test, y_true_genres, y_pred_genres):
    print(f"{file:<25} {asli:<15} {prediksi:<15}")
