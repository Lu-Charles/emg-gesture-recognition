"""Trial-separated open-set development protocol and transparent rejection baselines."""

from itertools import permutations
import numpy as np
from scipy.linalg import pinvh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

KNOWN = tuple(range(10, 17))  # Zero-based dataset labels; rest is 16.
ACTIVE = tuple(range(10, 16))
ORDERS = tuple(permutations((1, 2, 3)))


def feature_arrays(features, labels, ids):
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1 or not len(ids) or ids.min() < 0 or ids.max() >= len(features):
        raise ValueError("Invalid trial indices")
    x = np.asarray(features[ids]).reshape(-1, features.shape[-1])
    y = np.repeat(np.asarray(labels)[ids], features.shape[1])
    if not np.isfinite(x).all():
        raise ValueError("Unread/unpopulated features requested")
    return x, y


def protocol_rows(rows):
    """Day1 trials1:5 fit,6 threshold,7 independent score; preserve target split."""
    result = []
    for index, original in enumerate(rows):
        r = dict(original, corpus_index=index)
        if r["group"] != "development":
            raise ValueError("Only development participants permitted")
        known = int(r["class_index"]) in KNOWN
        if int(r["session"]) == 1:
            t = int(r["trial"])
            role = (
                (
                    "enroll_fit"
                    if t <= 5
                    else "source_threshold" if t == 6 else "source_score"
                )
                if known
                else ("source_score" if t == 7 else "unused")
            )
        else:
            role = (
                "target_score"
                if r["role"] == "scoring"
                else "target_calibration" if known else "unused"
            )
        r["open_role"] = role
        r["known"] = known
        result.append(r)
    return result


def select(rows, participant, session, role, ranks=None):
    return [
        r["corpus_index"]
        for r in rows
        if int(r["participant"]) == participant
        and int(r["session"]) == session
        and r["open_role"] == role
        and (ranks is None or int(r["calibration_rank"]) in ranks)
    ]


def allocations(rows, participant, session):
    source = select(rows, participant, 1, "enroll_fit")
    source_q = select(rows, participant, 1, "source_threshold")
    scoring = select(rows, participant, session, "target_score")
    for order in ORDERS:
        yield dict(
            method="none",
            budget=0,
            order=order,
            fit=source,
            threshold=source_q,
            score=scoring,
            target_fit=[],
        )
        for budget in (1, 2, 3):
            cal = select(
                rows, participant, session, "target_calibration", order[:budget]
            )
            yield dict(
                method="model_only",
                budget=budget,
                order=order,
                fit=source + cal,
                threshold=source_q,
                score=scoring,
                target_fit=cal,
            )
            yield dict(
                method="threshold_only",
                budget=budget,
                order=order,
                fit=source,
                threshold=cal,
                score=scoring,
                target_fit=[],
            )
            if budget >= 2:
                update = select(
                    rows,
                    participant,
                    session,
                    "target_calibration",
                    order[: budget - 1],
                )
                q = select(
                    rows,
                    participant,
                    session,
                    "target_calibration",
                    [order[budget - 1]],
                )
                yield dict(
                    method="both",
                    budget=budget,
                    order=order,
                    fit=source + update,
                    threshold=q,
                    score=scoring,
                    target_fit=update,
                )


def validate_allocation(a, rows):
    by_id = {r["corpus_index"]: r for r in rows}
    f, q, s = (set(a[k]) for k in ("fit", "threshold", "score"))
    if not f or not q or not s or f & q or f & s or q & s:
        raise ValueError("Empty or overlapping fit/threshold/score trials")
    if any(int(by_id[i]["class_index"]) not in KNOWN for i in f | q):
        raise ValueError("Unknown gesture used for fitting/calibration")
    if len({by_id[i]["participant"] for i in f | q | s}) != 1:
        raise ValueError("Participant mixing")
    target_used = {i for i in f | q if int(by_id[i]["session"]) > 1}
    if len(target_used) * 5 != a["budget"] * 35:
        raise ValueError("Recording budget mismatch")
    if set(a["target_fit"]) != {i for i in f if int(by_id[i]["session"]) > 1}:
        raise ValueError("Target-fit ledger mismatch")


def tdar_rms(x):
    """Feature-major MAV,ZC,SSC,WL,AR1..4,RMS; Burg librosa0.11.0, float64.

    Strict zero crossings; SSC nonnegative product includes flat triples.
    LPC denominator coefficients (not negated); no demeaning/normalization.
    All-zero rows explicitly receive zero AR coefficients.
    """
    import librosa

    if librosa.__version__ != "0.11.0":
        raise ValueError("Pinned librosa0.11.0 required")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] < 5 or not np.isfinite(x).all():
        raise ValueError("Finite windows,channels,samples array required")
    d = np.diff(x, axis=-1)
    mav = np.abs(x).mean(-1)
    zc = (
        ((x[..., :-1] > 0) & (x[..., 1:] < 0)) | ((x[..., :-1] < 0) & (x[..., 1:] > 0))
    ).sum(-1)
    ssc = (-d[..., :-1] * d[..., 1:] >= 0).sum(-1)
    wl = np.abs(d).sum(-1)
    flat = x.reshape(-1, x.shape[-1])
    nonzero = np.any(flat != 0, axis=-1)
    ar = np.zeros((len(flat), 4))
    ar[nonzero] = librosa.lpc(flat[nonzero], order=4, axis=-1)[:, 1:]
    ar = ar.reshape(*x.shape[:2], 4)
    rms = np.sqrt(np.mean(x * x, axis=-1))
    features = np.concatenate(
        [mav, zc, ssc, wl, *[ar[..., k] for k in range(4)], rms], axis=1
    )
    if not np.isfinite(features).all():
        raise FloatingPointError("Nonfinite TDAR; do not silently impute")
    return features


class LDADistance:
    """Equal-prior shrinkage LDA plus distance to its predicted class centroid.

    Reuses the identical fitted model for probability and Mahalanobis rejection;
    does not claim to implement Gao's learned prototype model.
    """

    def fit(self, x, y):
        if set(np.unique(y)) != set(KNOWN):
            raise ValueError("Fit must contain all seven known classes only")
        self.scaler = StandardScaler().fit(x)
        self.lda = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", priors=np.ones(7) / 7
        )
        self.lda.fit(self.scaler.transform(x), y)
        self.precision = pinvh(self.lda.covariance_)
        return self

    def score(self, x):
        z = self.scaler.transform(x)
        probs = self.lda.predict_proba(z)
        indices = probs.argmax(1)
        delta = z - self.lda.means_[indices]
        d2 = np.einsum("ni,ij,nj->n", delta, self.precision, delta, optimize=True)
        return self.lda.classes_[indices], {
            "probability": probs.max(1),
            "distance": -np.sqrt(np.maximum(d2, 0)),
        }


def cutoff(scores, retention=0.95):
    scores = np.asarray(scores)
    if not len(scores) or not np.isfinite(scores).all() or not 0 < retention <= 1:
        raise ValueError("Invalid calibration scores/retention")
    return float(np.quantile(scores, 1 - retention, method="lower"))


def measures(y, pred, scores, threshold):
    accepted = scores >= threshold
    active = np.isin(y, ACTIVE)
    unknown = y < 10
    rest = y == 16
    command = accepted & (pred != 16)

    def avg(mask, event):
        return float(np.mean(event[mask])) if mask.any() else None

    return dict(
        known_correct_acceptance=avg(active, accepted & (pred == y)),
        unknown_false_acceptance=avg(unknown, command),
        known_wrong_command=avg(active, command & (pred != y)),
        known_rejection=avg(active, ~accepted),
        known_predicted_rest=avg(active, accepted & (pred == 16)),
        rest_false_activation=avg(rest, command),
        closed_known_accuracy=avg(active, pred == y),
    )


def command_curve(y, pred, scores):
    """Exact tie-grouped descriptive CCR/UFA curve, never used to fit a cutoff."""
    active, unknown = np.isin(y, ACTIVE), y < 10
    order = np.argsort(-scores, kind="stable")
    values = scores[order]
    ends = np.r_[np.flatnonzero(values[1:] != values[:-1]), len(values) - 1]
    correct = np.cumsum((active & (pred == y))[order])[ends] / active.sum()
    false = np.cumsum((unknown & (pred != 16))[order])[ends] / unknown.sum()
    return np.r_[np.inf, values[ends]], np.r_[0.0, correct], np.r_[0.0, false]
