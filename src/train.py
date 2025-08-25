import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        # Empty audio check
        if audio.size == 0:
            print(f"⚠️ Skipped empty audio: {file_path}")
            return None

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        # No MFCC extracted
        if mfccs.shape[1] == 0:
            print(f"⚠️ Skipped no MFCC extracted: {file_path}")
            return None

        # Pad/truncate MFCCs safely
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        print(f"✅ Processed: {file_path}")  # success log
        return mfccs

    except Exception as e:
        print(f"❌ Error extracting {file_path}: {e}")
        return None


# ---------------- DATA LOADING ----------------
def load_dataset(dataset_path, limit_per_class=500):   # 👈 limit (e.g., 500 files per class)
    features, labels = [], []
    classes = os.listdir(dataset_path)

    for label, lang in enumerate(classes):
        lang_path = os.path.join(dataset_path, lang)
        if not os.path.isdir(lang_path):
            continue

        count = 0
        for file in os.listdir(lang_path):
            if count >= limit_per_class:   # 👈 Stop after limit
                break

            file_path = os.path.join(lang_path, file)
            mfccs = extract_features(file_path)
            if mfccs is not None:
                features.append(mfccs)
                labels.append(label)
                count += 1   # 👈 only increment when successfully added

        print(f"✅ Loaded {count} files from {lang}")

    return np.array(features), np.array(labels), classes


# ---------------- MODEL ----------------
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ---------------- MAIN ----------------
if __name__ == "__main__":
    dataset_path = "D:\\language classifier\\Indian-Language-Classifier\\data"

    print("🔄 Loading dataset...")
    X, y, classes = load_dataset(dataset_path, limit_per_class=2000)  # 👈 change limit if needed

    print(f"📊 Total valid samples: {len(X)}")
    print(f"📂 Classes found: {classes}")

    # reshape for CNN
    X = X[..., np.newaxis]

    # split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🔄 Building model...")
    model = build_model(X_train.shape[1:], len(classes))

    print("🚀 Training started...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    print("💾 Saving model to language_classifier.h5")
    model.save("language_classifier.h5")

    
