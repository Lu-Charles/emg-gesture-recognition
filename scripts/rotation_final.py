"""Development falsification of gain-profiled rotation and source-only pair selection."""

import itertools
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.senic import rotate_features, estimate_rotation, DEVELOPMENT
from src.grabmyo_corpus import hash_stream
from threadpoolctl import threadpool_limits
from src.final_coverage import validate_senic as validate_split
from scripts.cache_final_coverage import frozen

COHORT = frozen()["senic"]["participants"]
R = Path(__file__).resolve().parents[1]
O = R / "research/runs/20260916_rotation_final"
C = R / "data/public/senic/cache_final_20260916"


def profile_rotation(a, b, fractional=True):
    """Fit a shared channel gain at each candidate shift, then choose the lowest residual.

    a contains source templates; b contains target templates. Ties prefer
    smaller absolute shifts. One template cannot identify both shift and gain,
    so the one-template case uses zero shift."""
    shifts = sorted(
        np.arange(-4.0, 4.0, 0.125 if fractional else 1),
        key=lambda shift: (abs(shift), shift),
    )
    if len(a) == 1:
        shifts = [0.0]  # all shifts exactly fit one positive template with free gains
    fits = []
    for shift in shifts:
        rotated_target = rotate_features(b, shift)
        log_ratios = np.log(np.maximum(a, 1e-12)) - np.log(
            np.maximum(rotated_target, 1e-12)
        )
        log_gain = log_ratios.mean(0)
        fits.append(
            (
                np.mean((log_ratios - log_gain) ** 2),
                shift,
                np.clip(np.exp(log_gain), 0.5, 2),
            )
        )
    best = min(range(len(fits)), key=lambda j: fits[j][0])
    cost, shift, gain = fits[best]
    return float(shift), gain, float(cost)


def selection(template):
    """Choose gesture pairs from source templates, without target-session access."""
    normalized_templates = template / np.maximum(
        np.linalg.norm(template, axis=1, keepdims=True), 1e-12
    )
    cosine_distance = 1 - normalized_templates @ normalized_templates.T
    np.fill_diagonal(cosine_distance, -np.inf)
    margin = np.full((7, 7), -np.inf)
    for g, h in itertools.permutations(range(7), 2):
        log_ratio = np.log(np.maximum(template[g], 1e-12)) - np.log(
            np.maximum(template[h], 1e-12)
        )
        margin[g, h] = min(
            np.mean((log_ratio - np.roll(log_ratio, s)) ** 2) for s in range(1, 8)
        )
    return dict(
        identifiable=list(map(int, np.unravel_index(margin.argmax(), margin.shape))),
        diverse=list(
            map(int, np.unravel_index(cosine_distance.argmax(), cosine_distance.shape))
        ),
        active=np.argsort(-template.sum(1), kind="stable")[:2].tolist(),
        fixed=[0, 1],
        margin=margin.tolist(),
    )


def synthetic():
    rng = np.random.default_rng(20260916)
    u = np.exp(rng.normal(0, 0.5, (2, 8)))
    gain = np.exp(rng.uniform(-0.3, 0.3, 8))
    target = np.roll(u / gain, -2, axis=-1)
    s, g, c = profile_rotation(u, target, False)
    assert s == 2 and np.allclose(g, gain) and c < 1e-25
    s1, g1, c1 = profile_rotation(u[:1], target[:1], False)
    assert s1 == 0 and c1 < 1e-25
    # unbounded analytical fit, distinct from clipped engineering adapter
    d = u[0] / target[0]
    error = np.linalg.norm(target[1] * d - u[1])
    assert error > 1e-2
    noisy = target * np.exp(rng.normal(0, 0.5, target.shape))
    sn, gn, cn = profile_rotation(u, noisy, False)
    return dict(
        scope="synthetic only",
        true_shift=2,
        two_template_shift=s,
        two_template_cost=c,
        single_template_shift=s1,
        single_unbounded_other_template_error=float(error),
        class_specific_gain_misspecification_shift=sn,
        misspecification_cost=cn,
    )


def main():
    O.mkdir(exist_ok=False)
    rows = json.loads((C / "trial_manifest.json").read_text())
    amp = np.load(C / "amplitude_features.npy")
    x = np.load(C / "tdar_rms_features.npy")
    assert x.shape == (5544, 15, 72)
    config = dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        scope="frozen independent evaluation; no final-score selection",
        cache=str(C),
        protocol_sha256=hash_stream(
            R / "research/runs/20260625_confirmatory_protocol/protocol.json"
        ),
        participants=list(COHORT),
        source="position0 repetitions0,1",
        calibration="two trials: g repetition0,h repetition1; K1 sameg; K2 all42ordered distinct pairs",
        score="position1..10 repetition2; all7classes;15windows; one vote/trial",
        budget="equal two recorded trials; actual lengths differ; four analyzed seconds",
        selection="maximize minimum squared separation of source log-RMS gesture ratio from its seven nonidentity integer rotations; no target access; deterministic first tie",
        profile="minimize shared-log-gain-profiled RMS residual over circular shifts; estimate gains then clip .5..2; oneclass ambiguous so shift0",
        methods=[
            "frozen",
            "pooled",
            "cosine_rotation",
            "cosine_gain",
            "profile_integer",
            "profile_fractional",
            "gain_only",
        ],
        angles="never read",
        code_sha256=hash_stream(__file__),
        cache_sha256=hash_stream(C / "tdar_rms_features.npy"),
    )
    (O / "protocol.json").write_text(json.dumps(config, indent=2) + "\n")
    (O / "rotation_final.py").write_bytes(Path(__file__).read_bytes())
    (O / "synthetic.json").write_text(json.dumps(synthetic(), indent=2) + "\n")
    records = []
    selections = []
    access = []
    predictions = {}
    parameters = {}
    start = time.perf_counter()

    def vote(z):
        return np.array(
            [np.bincount(r, minlength=7).argmax() for r in z.reshape(-1, 15)]
        )

    for p in COHORT:
        src = [
            i
            for i, r in enumerate(rows)
            if r["subject"] == p and r["position"] == 0 and r["repetition"] in [0, 1]
        ]
        sy = np.repeat([rows[i]["label"] for i in src], 15)
        sx = x[src].reshape(-1, 72)
        sc = StandardScaler().fit(sx)
        lda = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", priors=np.ones(7) / 7
        ).fit(sc.transform(sx), sy)
        template = np.array([sx[sy == g, 64:72].mean(0) for g in range(7)])
        selections.append(
            dict(
                participant=p, **selection(template), source_templates=template.tolist()
            )
        )
        for key, val in [
            ("mean", sc.mean_),
            ("scale", sc.scale_),
            ("coef", lda.coef_),
            ("intercept", lda.intercept_),
        ]:
            parameters[f"p{p}_{key}"] = val
        for pos in range(1, 11):
            pool = {
                (r["label"], r["repetition"]): i
                for i, r in enumerate(rows)
                if r["subject"] == p and r["position"] == pos
            }
            score = [pool[g, 2] for g in range(7)]
            tx = x[score].reshape(-1, 72)
            truth = np.arange(7)
            for g, h in itertools.product(range(7), repeat=2):
                cal = [pool[g, 0], pool[h, 1]]
                validate_split(src, cal, score, rows)
                gs = sorted(set([g, h]))
                cy = np.repeat([g, h], 15)
                cx = x[cal].reshape(-1, 72)
                a = template[gs]
                b = np.array([cx[cy == k, 64:72].mean(0) for k in gs])
                name = f"p{p}_pos{pos}_g{g}_h{h}"
                access.append(dict(name=name, source=src, calibration=cal, score=score))
                shift, cost = estimate_rotation(a, b)
                cg = np.clip(
                    np.exp(
                        np.mean(
                            np.log(np.maximum(a, 1e-12))
                            - np.log(np.maximum(rotate_features(b, shift), 1e-12)),
                            axis=0,
                        )
                    ),
                    0.5,
                    2,
                )
                gain = np.clip(
                    np.exp(
                        np.mean(
                            np.log(np.maximum(a, 1e-12)) - np.log(np.maximum(b, 1e-12)),
                            axis=0,
                        )
                    ),
                    0.5,
                    2,
                )
                pi, gi, ci = profile_rotation(a, b, False)
                pf, gf, cf = profile_rotation(a, b, True)
                methods = {
                    "frozen": (0.0, np.ones(8), 0.0),
                    "gain_only": (0.0, gain, 0.0),
                    "cosine_rotation": (shift, np.ones(8), cost),
                    "cosine_gain": (shift, cg, cost),
                    "profile_integer": (pi, gi, ci),
                    "profile_fractional": (pf, gf, cf),
                }
                pooled = LinearDiscriminantAnalysis(
                    solver="lsqr", shrinkage="auto", priors=np.ones(7) / 7
                ).fit(sc.transform(np.concatenate([sx, cx])), np.r_[sy, cy])
                parameters[name + "_pooled_coef"] = pooled.coef_
                parameters[name + "_pooled_intercept"] = pooled.intercept_
                for method in config["methods"]:
                    if method == "pooled":
                        z = pooled.predict(sc.transform(tx))
                        state = None
                    else:
                        s, ga, cost = methods[method]
                        corrected = rotate_features(tx, s)
                        corrected[:, np.r_[0:8, 24:32, 64:72]] *= np.tile(ga, 3)
                        z = lda.predict(sc.transform(corrected))
                        state = dict(shift=s, gain=ga.tolist(), cost=cost)
                    v = vote(z)
                    other = ~np.isin(truth, gs)
                    key = name + "_" + method
                    predictions[key] = v
                    records.append(
                        dict(
                            name=key,
                            participant=p,
                            position=pos,
                            g=g,
                            h=h,
                            k=len(gs),
                            method=method,
                            accuracy=float(np.mean(v == truth)),
                            window_accuracy=float(np.mean(z == np.repeat(truth, 15))),
                            other_accuracy=float(np.mean(v[other] == truth[other])),
                            other_into_calibration=float(
                                np.mean(np.isin(v[other], gs))
                            ),
                            calibrated_recall=float(
                                np.mean(v[~other] == truth[~other])
                            ),
                            recorded_seconds=sum(
                                rows[i]["recorded_seconds"] for i in cal
                            ),
                            state=state,
                        )
                    )
        (O / "results.json").write_text(
            json.dumps(
                dict(config=config, records=records, selections=selections), indent=2
            )
            + "\n"
        )
        (O / "access.json").write_text(json.dumps(access, indent=2) + "\n")
        np.savez_compressed(O / "predictions.npz", **predictions)
        np.savez_compressed(O / "parameters.npz", **parameters)
        print(
            json.dumps(
                dict(
                    participant=p,
                    records=len(records),
                    seconds=time.perf_counter() - start,
                )
            ),
            flush=True,
        )
    (O / "complete.json").write_text(
        json.dumps(
            dict(
                records=len(records),
                cases=len(access),
                seconds=time.perf_counter() - start,
            )
        )
        + "\n"
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
