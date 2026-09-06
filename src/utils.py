from __future__ import annotations
import os
from pathlib import Path
import time

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def ts_id(prefix: str = "") -> str:
    s = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}{s}" if prefix else s