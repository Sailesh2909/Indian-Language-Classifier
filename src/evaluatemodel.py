import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        if audio.size == 0:
            return None
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print(f"❌ Error with {file_path}: {e}")
        return None

# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Load model
    model = tf.keras.models.load_model(r"D:\language classifier\Indian-Language-Classifier\language_classifier.h5")

    # Load test dataset
    dataset_path = r"D:\language classifier\Indian-Language-Classifier\data"
    classes = os.listdir(dataset_path)

    X_test, y_test = [], []
    for label, lang in enumerate(classes):
        lang_path = os.path.join(dataset_path, lang)
        for file in os.listdir(lang_path)[:20]:
            mfccs = extract_features(os.path.join(lang_path, file))
            if mfccs is not None:
                X_test.append(mfccs)
                y_test.append(label)

    X_test = np.array(X_test)[..., np.newaxis]
    y_test = np.array(y_test)

    # Evaluate
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy on Test Subset: {acc*100:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
