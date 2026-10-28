"""Fetch and verify pinned official inputs; never imported by the runtime viewer."""

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/official"
LOCK = ROOT / "models/scout_piper/upstream_lock.json"


def ensure_sources(fetch=False):
    lock = json.loads(LOCK.read_text())
    for key, source in lock["repositories"].items():
        for relative, digest in source["files"].items():
            target = CACHE / key / relative
            if not target.exists():
                if not fetch:
                    raise FileNotFoundError(
                        f"{target}; run scripts/build_model.py --fetch"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                url = source["repository"].replace(
                    "https://github.com/", "https://raw.githubusercontent.com/"
                )
                urllib.request.urlretrieve(
                    f"{url}/{source['commit']}/{relative}", target
                )
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Pinned input checksum mismatch: {target}")
    return lock
