"""Fill only development feature rows absent from the validated TDAR cache."""

import json
import time
from pathlib import Path
import numpy as np
from src.grabmyo_corpus import WindowCorpus, hash_stream
from src.open_set_emg import tdar_rms

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/runs/20260916_tdar_development"


def main():
    OUT.mkdir(exist_ok=False)
    corpus = WindowCorpus(ROOT / "data/public/grabmyo/cache_20260906_v1", "development")
    old = ROOT / "research/runs/20260906_open_set_classical_v2"
    oldrows = json.loads((old / "trial_manifest.json").read_text())
    assert [r["record"] for r in oldrows] == [r["record"] for r in corpus.rows]
    previous = np.load(old / "features.npy", mmap_mode="r")
    features = np.lib.format.open_memmap(
        OUT / "features.npy", mode="w+", dtype="float64", shape=previous.shape
    )
    features[:] = previous
    missing = np.flatnonzero(~np.isfinite(previous).all((1, 2))).tolist()
    (OUT / "protocol.json").write_text(
        json.dumps(
            dict(
                scope="development feature extraction only",
                missing=missing,
                source=str(old),
                source_sha256=hash_stream(old / "features.npy"),
                manifest_sha256=hash_stream(corpus.path / "development_manifest.csv"),
                code_sha256=hash_stream(__file__),
                feature_code_sha256=hash_stream(ROOT / "src/open_set_emg.py"),
            ),
            indent=2,
        )
        + "\n"
    )
    start = time.perf_counter()
    for offset in range(0, len(missing), 16):
        ids = missing[offset : offset + 16]
        x, _ = corpus.batch(corpus.window_ids(ids))
        features[ids] = tdar_rms(x).reshape(len(ids), 35, 144)
        if offset % 256 == 0:
            print(offset, len(missing), flush=True)
    features.flush()
    assert np.isfinite(features).all()
    unchanged = np.flatnonzero(np.isfinite(previous).all((1, 2)))
    np.testing.assert_array_equal(features[unchanged], previous[unchanged])
    (OUT / "complete.json").write_text(
        json.dumps(
            dict(
                filled=len(missing),
                reused=len(unchanged),
                seconds=time.perf_counter() - start,
                output_sha256=hash_stream(OUT / "features.npy"),
                finite=True,
            ),
            indent=2,
        )
        + "\n"
    )
    (OUT / "complete_tdar_development.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
