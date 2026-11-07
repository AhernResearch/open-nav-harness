"""Fingerprint executable source and prompts, independent of Git staging."""

from hashlib import sha256
from pathlib import Path


# No cache: freezing and launching must detect edits made while the API is up.
def implementation_digest() -> str:
    root = Path(__file__).resolve().parents[2]
    digest = sha256()
    for package in ("nav_harness", "navgpt"):
        for path in sorted((root / package).rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()
