"""Retrieve only frozen SeNic final archives and verify pinned Git blob identities."""

import argparse
import hashlib
import json
from pathlib import Path
import requests
from scripts.cache_final_coverage import frozen

R = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-existing", action="store_true")
    a = ap.parse_args()
    cfg = frozen()
    ledger = json.loads(
        (
            R
            / "research/runs/20260907_sensor_shift_audit/download_tree_verification.json"
        ).read_text()
    )
    root = R / "data/public/SeNic"
    checked = []
    assert ledger["current_commit"] == "a4c12f7daab28a80d557677ae8dbcef0d7871ba2"
    for p in cfg["senic"]["participants"]:
        entries = [
            z
            for z in ledger["checked_files"]
            if z["path"] in [f"h{p}.rar", f"h{p}/0-4.rar"]
        ]
        assert len(entries) == 1
        entry = entries[0]
        target = root / entry["path"]
        url = f"https://raw.githubusercontent.com/BoZhuBo/SeNic/{ledger['current_commit']}/{entry['path']}"
        if target.exists():
            data = target.read_bytes()
        else:
            if a.verify_existing:
                raise FileNotFoundError(target)
            response = requests.get(url, timeout=(15, 120))
            response.raise_for_status()
            data = response.content
        blob = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\x00" + data
        ).hexdigest()
        assert len(data) == entry["bytes"] and blob == entry["git_blob_sha1"], target
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".partial")
            tmp.write_bytes(data)
            tmp.replace(target)
        checked.append(
            dict(
                path=entry["path"],
                git_blob_sha1=blob,
                sha256=hashlib.sha256(data).hexdigest(),
                url=url,
            )
        )
    print(
        json.dumps(
            dict(
                passed=True,
                archives=len(checked),
                mode="verify existing" if a.verify_existing else "retrieve and verify",
                checked=checked,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
