import librosa
import numpy as np

def extract_features(file_path, max_pad_len=100):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
        return mfccs
    except Exception as e:
        print("Error extracting features:", e)
        return None
