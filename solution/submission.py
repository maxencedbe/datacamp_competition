import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

import warnings
warnings.filterwarnings("ignore")

IMG_SIZE = (32, 32)


def load_images(df, img_dir, size=IMG_SIZE):
    """Load images and flatten them into feature vectors."""
    features = []
    for fname in df["filename"]:
        path = os.path.join(img_dir, fname)
        try:
            img = Image.open(path).convert("RGB").resize(size)
            arr = np.array(img, dtype=np.float32).flatten() / 255.0
        except Exception:
            arr = np.zeros(size[0] * size[1] * 3, dtype=np.float32)
        features.append(arr)
    return np.array(features)


class Model:
    def __init__(self):
        self.clf = OneVsRestClassifier(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1,
            )
        )

    def fit(self, X_train, y_train, train_img_dir):
        print("Loading training images...")
        X = load_images(X_train, train_img_dir)
        y = y_train.values
        print("Training Random Forest ({} samples, {} features)...".format(
            X.shape[0], X.shape[1]
        ))
        self.clf.fit(X, y)
        print("Training done.")

    def predict(self, X_test, test_img_dir):
        print("Loading test images...")
        X = load_images(X_test, test_img_dir)
        print("Generating predictions...")
        proba = self.clf.predict_proba(X)
        # predict_proba with OneVsRestClassifier returns list of arrays
        # Extract probability of class 1 for each label
        if isinstance(proba, list):
            result = np.column_stack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in proba])
        else:
            result = proba
        return result


def get_model():
    return Model()