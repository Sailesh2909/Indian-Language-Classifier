import numpy as np
import librosa

def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')  

        # Check for empty audio
        if audio.size == 0:
            print(f"⚠️ Empty audio file skipped: {file_path}")
            return None

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        # Check for empty MFCCs
        if mfccs.shape[1] == 0:
            print(f"⚠️ No MFCC extracted: {file_path}")
            return None

        # Pad/truncate MFCCs safely
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        return mfccs

    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None
