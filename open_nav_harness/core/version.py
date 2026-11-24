"""Fingerprint executable source and prompts, independent of Git staging."""

from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def source_digest(paths) -> str:
    digest = sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def packages(*names) -> set[Path]:
    return {path for name in names for path in (ROOT / name).rglob("*.py")}


# No cache: freezing and launching must detect edits made while the API is up.
def kernel_digest() -> str:
    """Everything a kernel run executes: tools, prompt, loop, provider and context code."""
    return source_digest(packages("open_nav_harness/kernel", "open_nav_harness/core",
                                  "open_nav_harness/providers"))
