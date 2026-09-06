# from __future__ import annotations
# import joblib
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler

# def make_rf(seed: int = 42) -> Pipeline:
#     # scaler helps when you mix spectral + time features
#     clf = RandomForestClassifier(
#         n_estimators=400,
#         random_state=seed,
#         class_weight="balanced",
#         n_jobs=-1,
#         max_depth=None,
#         min_samples_leaf=2,
#     )
#     return Pipeline([("scaler", StandardScaler()), ("rf", clf)])

# def save_model(model, path: str):
#     joblib.dump(model, path)

# def load_model(path: str):
#     return joblib.load(path)

from __future__ import annotations
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def make_rf(seed: int = 42) -> Pipeline:
    # NOTE: scaler is not necessary for RandomForest, but keeping it is fine
    clf = RandomForestClassifier(
        n_estimators=200,        # fewer trees
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,

        # key anti-overfit controls
        max_depth=10,            # limit memorization
        min_samples_leaf=5,      # smoother boundaries
        min_samples_split=10,
        max_features="sqrt",     # more randomness per split
        bootstrap=True,
    )
    return Pipeline([("scaler", StandardScaler()), ("rf", clf)])

def save_model(model, path: str):
    joblib.dump(model, path)

def load_model(path: str):
    return joblib.load(path)
