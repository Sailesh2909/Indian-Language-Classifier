import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from preprocess import extract_features
from model import create_model

DATASET_DIR = "D:\\language classifier\\Indian-Language-Classifier\\data"
languages = os.listdir(DATASET_DIR)  # each folder = one language

X, y = [], []
for idx, lang in enumerate(languages):
    files = os.listdir(os.path.join(DATASET_DIR, lang))
    for f in files:
        feature = extract_features(os.path.join(DATASET_DIR, lang, f))
        if feature is not None:
            X.append(feature)
            y.append(idx)

X = np.array(X)
y = np.array(y)

# reshape for CNN
X = X[..., np.newaxis]
y = to_categorical(y, num_classes=len(languages))

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# build model
model = create_model(X_train[0].shape, len(languages))

# train
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# save model
model.save("indian_language_model.h5")
