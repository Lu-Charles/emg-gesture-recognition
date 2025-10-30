from __future__ import annotations
import numpy as np
from collections import deque

class SelfCalibrator:
    def __init__(self, max_buf: int = 2000, conf_thresh: float = 0.90):
        self.max_buf = max_buf
        self.conf_thresh = conf_thresh
        self.Xb = deque(maxlen=max_buf)
        self.yb = deque(maxlen=max_buf)

    def ingest(self, model, X: np.ndarray):
        # needs predict_proba
        proba = model.predict_proba(X)
        yhat = np.argmax(proba, axis=1)
        conf = np.max(proba, axis=1)
        for xi, yi, ci in zip(X, yhat, conf):
            if ci >= self.conf_thresh:
                self.Xb.append(xi.astype(np.float32))
                self.yb.append(int(yi))

    def has_enough(self, n: int = 200) -> bool:
        return len(self.Xb) >= n

    def adapt(self, model, X_seed: np.ndarray, y_seed: np.ndarray):
        if len(self.Xb) == 0:
            return model
        Xb = np.stack(list(self.Xb), axis=0)
        yb = np.array(list(self.yb))
        X_new = np.concatenate([X_seed, Xb], axis=0)
        y_new = np.concatenate([y_seed, yb], axis=0)
        model.fit(X_new, y_new)
        return model