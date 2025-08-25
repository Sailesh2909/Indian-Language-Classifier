import numpy as np
import librosa
import tensorflow as tf
import streamlit as st
import os

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
        st.error(f"❌ Error extracting features: {e}")
        return None

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "language_classifier.h5"  # or "language_classifier.tflite"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Hardcoded class labels (since we won’t ship the full dataset)
classes = [
    "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam",
    "Marathi", "Punjabi", "Tamil", "Telugu", "Urdu"
]

# ---------------- STREAMLIT UI ----------------
st.title("🌐 Indian Language Classifier")
st.markdown("Upload an audio file and let the model predict the language!")

uploaded_file = st.file_uploader("🎵 Upload an audio file (.wav/.mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("temp_audio.wav")

    mfccs = extract_features("temp_audio.wav")
    if mfccs is not None:
        mfccs = np.expand_dims(mfccs, axis=(0, -1))
        prediction = np.argmax(model.predict(mfccs), axis=1)[0]
        st.success(f"🔮 Predicted Language: **{classes[prediction]}**")
