"""Fetch only train/development GRABMyo records and build versioned trial caches."""
import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from src.grabmyo import checksum_index, read_forearm, short_windows, amplitude_features
from src.grabmyo_corpus import fetch_s3, hash_stream, permitted_rows, S3_URL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/public/grabmyo/1.1.0"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("Use 1–16 download workers")
    args.out.mkdir(parents=True, exist_ok=False)
    rows = permitted_rows(args.preparation)
    config = {"created_utc": datetime.now(timezone.utc).isoformat(), "dataset": "GRABMyo1.1.0",
              "groups": ["train", "development"], "preparation": str(args.preparation.resolve()),
              "manifest_sha256": hash_stream(args.preparation / "trial_manifest.csv"),
              "checksum_list_sha256": hash_stream(args.data / "SHA256SUMS.txt"),
              "base_url": S3_URL, "download_workers": args.workers,
              "signals_dtype": "float32", "signals_layout": "trial,channel,sample",
              "features_dtype": "float64", "channels": "F1–F16 physical mV", "fs": 2048,
              "window_samples": 512, "hop_samples": 256, "first_start_sample": 1024,
              "code_hashes": {f: hash_stream(f) for f in ("src/grabmyo.py", "src/grabmyo_corpus.py", "scripts/cache_grabmyo.py")}}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    checksums = checksum_index(args.data)
    start = time.perf_counter()
    files = [r["record"] + suffix for r in rows for suffix in (".hea", ".dat")]
    done, downloaded_bytes, failures = 0, 0, []
    try:
        with (args.out / "retrieval.jsonl").open("w") as log, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch_s3, args.data, name, checksums): name for name in files}
            for future in as_completed(futures):
                try:
                    item = future.result()
                except Exception as exc:
                    item = {"path": futures[future], "error": str(exc)}
                    failures.append(item)
                log.write(json.dumps(item) + "\n")
                done += 1
                if not item.get("cached", True):
                    downloaded_bytes += item["bytes"]
                if done % 500 == 0:
                    log.flush()
                    progress = {"phase": "download", "files": done, "total_files": len(files),
                                "new_MB": downloaded_bytes / 1e6, "seconds": time.perf_counter() - start,
                                "failures": len(failures)}
                    (args.out / "progress.json").write_text(json.dumps(progress))
                    print(json.dumps(progress), flush=True)
        if failures:
            raise RuntimeError(f"{len(failures)} files failed; see retrieval.jsonl")
        download_seconds = time.perf_counter() - start
        group_results = {}
        for group in ("train", "development"):
            subset = [r for r in rows if r["group"] == group]
            with (args.out / f"{group}_manifest.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(subset[0]))
                writer.writeheader(); writer.writerows(subset)
            signals = np.lib.format.open_memmap(args.out / f"{group}_signals.npy", mode="w+", dtype="float32",
                                                 shape=(len(subset), 16, 10240))
            features = np.lib.format.open_memmap(args.out / f"{group}_features.npy", mode="w+", dtype="float64",
                                                  shape=(len(subset), 35, 48))
            max_error = 0.
            with (args.out / f"{group}_signal_checks.jsonl").open("w") as log:
                for index, row in enumerate(subset):
                    x, info = read_forearm(args.data, row["record"], checksums)
                    signals[index] = x.T.astype(np.float32)
                    windows, _ = short_windows(x)
                    features[index] = amplitude_features(windows)
                    max_error = max(max_error, info["manual_decode_max_abs_error"])
                    log.write(json.dumps({"record": row["record"], **info}) + "\n")
                    if (index + 1) % 500 == 0:
                        print(json.dumps({"phase": "decode", "group": group, "trials": index + 1,
                                          "total": len(subset)}), flush=True)
            signals.flush(); features.flush()
            del signals, features
            group_results[group] = {"trials": len(subset), "windows": len(subset) * 35,
                                    "participants": sorted({int(r["participant"]) for r in subset}),
                                    "manual_decode_max_abs_error": max_error,
                                    "signals_sha256": hash_stream(args.out / f"{group}_signals.npy"),
                                    "features_sha256": hash_stream(args.out / f"{group}_features.npy"),
                                    "manifest_sha256": hash_stream(args.out / f"{group}_manifest.csv")}
        summary = {"completed_utc": datetime.now(timezone.utc).isoformat(), "groups": group_results,
                   "files_verified": done, "downloaded_bytes": downloaded_bytes,
                   "download_seconds": download_seconds, "total_seconds": time.perf_counter() - start,
                   "final_participants_accessed": 0, "failed_files": len(failures)}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        snapshot = args.out / "code_snapshot"; snapshot.mkdir()
        for name in config["code_hashes"]:
            (snapshot / Path(name).name).write_bytes(Path(name).read_bytes())
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        (args.out / "failure.json").write_text(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "error": str(exc)}))
        raise


if __name__ == "__main__":
    main()
