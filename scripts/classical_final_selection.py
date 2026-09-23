"""Exhaustive two-recording classical controls and source-only selection screen."""

import itertools
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from src.grabmyo_corpus import hash_stream
from src.final_coverage import FinalCorpus as WindowCorpus

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/runs/20260916_classical_final"


def main():
    OUT.mkdir(exist_ok=False)
    corpus = WindowCorpus(ROOT / "data/public/grabmyo/cache_final_20260916", "final")
    rows = corpus.rows
    tdar = np.load(
        ROOT / "research/runs/20260916_tdar_final/features.npy", mmap_mode="r"
    )
    assert np.isfinite(tdar).all()
    people = sorted({int(r["participant"]) for r in rows})
    config = dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        scope="frozen independent evaluation",
        protocol_sha256=hash_stream(
            ROOT / "research/runs/20260625_confirmatory_protocol/protocol.json"
        ),
        participants=people,
        calibration="K1 g ranks1,2; K2 all240 ordered distinct pairs g rank1,h rank2",
        seconds=10,
        methods=["amplitude_gain48", "tdar_frozen", "tdar_gain16", "tdar_pooled"],
        source_selection="cosine distance between source class RMS templates; select most diverse ordered pair; deterministic first tie",
        source_selection_other="largest sum of RMS amplitudes; pair sorted by descending source activity, no target access",
        main="random-pair expectation across all240 pairs; proposed selectors exploratory",
        score="same68 target trials and35windows;17classes; lowest-index vote tie",
        code_sha256=hash_stream(__file__),
        tdar_sha256=hash_stream(
            ROOT / "research/runs/20260916_tdar_final/features.npy"
        ),
    )
    (OUT / "protocol.json").write_text(json.dumps(config, indent=2) + "\n")
    (OUT / "coverage_classical_selection.py").write_bytes(Path(__file__).read_bytes())
    saved = {}
    parameters = {}
    records = []
    selections = []
    access = []
    start = time.perf_counter()

    def vote(a):
        return np.array(
            [np.bincount(z, minlength=17).argmax() for z in a.reshape(-1, 35)]
        )

    for p in people:
        source = [
            i
            for i, r in enumerate(rows)
            if int(r["participant"]) == p and r["role"] == "enrollment"
        ]
        sy = np.repeat([int(rows[i]["class_index"]) for i in source], 35)
        ax = np.asarray(corpus.features[source]).reshape(-1, 48)
        dx = np.asarray(tdar[source]).reshape(-1, 144)
        asc = StandardScaler().fit(ax)
        dsc = StandardScaler().fit(dx)
        ala = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", priors=np.full(17, 1 / 17)
        ).fit(asc.transform(ax), sy)
        dla = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", priors=np.full(17, 1 / 17)
        ).fit(dsc.transform(dx), sy)
        amean = np.array([ax[sy == g].mean(0) for g in range(16)])
        template = amean[:, 16:32]
        unit = template / np.maximum(
            np.linalg.norm(template, axis=1, keepdims=True), 1e-12
        )
        distances = 1 - unit @ unit.T
        np.fill_diagonal(distances, -np.inf)
        diverse = list(map(int, np.unravel_index(distances.argmax(), distances.shape)))
        active = np.argsort(-template.sum(1), kind="stable")[:2].tolist()
        selections.append(
            dict(
                participant=p,
                diverse=diverse,
                active=active,
                fixed=[0, 1],
                source_rms_templates=template.tolist(),
            )
        )
        parameters[f"p{p}_amplitude_coef"] = ala.coef_
        parameters[f"p{p}_amplitude_intercept"] = ala.intercept_
        parameters[f"p{p}_amplitude_mean"] = asc.mean_
        parameters[f"p{p}_amplitude_scale"] = asc.scale_
        parameters[f"p{p}_tdar_coef"] = dla.coef_
        parameters[f"p{p}_tdar_intercept"] = dla.intercept_
        parameters[f"p{p}_tdar_mean"] = dsc.mean_
        parameters[f"p{p}_tdar_scale"] = dsc.scale_
        for session in [2, 3]:
            pool = {
                (int(r["class_index"]), int(r["calibration_rank"])): i
                for i, r in enumerate(rows)
                if int(r["participant"]) == p
                and int(r["session"]) == session
                and r["role"] == "calibration"
            }
            score = [
                i
                for i, r in enumerate(rows)
                if int(r["participant"]) == p
                and int(r["session"]) == session
                and r["role"] == "scoring"
            ]
            truth = np.array([int(rows[i]["class_index"]) for i in score])
            tx = np.asarray(tdar[score]).reshape(-1, 144)
            ta = np.asarray(corpus.features[score]).reshape(-1, 48)
            frozen = vote(dla.predict(dsc.transform(tx)))
            for g, h in itertools.product(range(16), repeat=2):
                k = 1 if g == h else 2
                cal = [pool[g, 1], pool[h, 2]]
                gestures = sorted(set([g, h]))
                name = f"p{p}_s{session}_g{g}_h{h}"
                assert not set(cal) & set(score) and not set(cal) & set(source)
                access.append(
                    dict(name=name, source=source, calibration=cal, score=score)
                )
                cy = np.repeat([g, h], 35)
                cx = np.asarray(tdar[cal]).reshape(-1, 144)
                ca = np.asarray(corpus.features[cal]).reshape(-1, 48)
                gain = np.clip(
                    np.exp(
                        np.mean(
                            [
                                np.log(np.maximum(amean[c], 1e-12))
                                - np.log(np.maximum(ca[cy == c].mean(0), 1e-12))
                                for c in gestures
                            ],
                            axis=0,
                        )
                    ),
                    0.5,
                    2,
                )
                coherent = np.tile(gain[16:32], 3)
                corrected = tx.copy()
                for sl in [slice(0, 16), slice(48, 64), slice(128, 144)]:
                    corrected[:, sl] *= gain[16:32]
                pooled = LinearDiscriminantAnalysis(
                    solver="lsqr", shrinkage="auto", priors=np.full(17, 1 / 17)
                ).fit(dsc.transform(np.concatenate([dx, cx])), np.r_[sy, cy])
                parameters[name + "_pooled_coef"] = pooled.coef_
                parameters[name + "_pooled_intercept"] = pooled.intercept_
                parameters[name + "_gain"] = gain
                predictions = {
                    "amplitude_gain48": vote(ala.predict(asc.transform(ta * gain))),
                    "tdar_frozen": frozen,
                    "tdar_gain16": vote(dla.predict(dsc.transform(corrected))),
                    "tdar_pooled": vote(pooled.predict(dsc.transform(tx))),
                }
                for method, v in predictions.items():
                    selected = np.isin(truth, gestures)
                    other = (truth < 16) & ~selected
                    key = name + "_" + method
                    saved[key] = v
                    records.append(
                        dict(
                            name=key,
                            participant=p,
                            session=session,
                            g=g,
                            h=h,
                            k=k,
                            method=method,
                            accuracy=float(np.mean(v == truth)),
                            macro_f1=float(
                                f1_score(
                                    truth,
                                    v,
                                    labels=range(17),
                                    average="macro",
                                    zero_division=0,
                                )
                            ),
                            other_active_accuracy=float(
                                np.mean(v[other] == truth[other])
                            ),
                            other_into_calibration=float(
                                np.mean(np.isin(v[other], gestures))
                            ),
                        )
                    )
            print(
                json.dumps(
                    dict(
                        participant=p,
                        session=session,
                        records=len(records),
                        seconds=time.perf_counter() - start,
                    )
                ),
                flush=True,
            )
            (OUT / "results.json").write_text(
                json.dumps(
                    dict(config=config, records=records, selections=selections),
                    indent=2,
                )
                + "\n"
            )
            (OUT / "access.json").write_text(json.dumps(access, indent=2) + "\n")
            np.savez_compressed(OUT / "predictions.npz", **saved)
            np.savez_compressed(OUT / "parameters.npz", **parameters)
    (OUT / "complete.json").write_text(
        json.dumps(dict(records=len(records), seconds=time.perf_counter() - start))
        + "\n"
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
