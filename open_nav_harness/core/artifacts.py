"""Lossless content-addressed media and replayable request journal."""

import base64
import hashlib
import json
import os
import time
from pathlib import Path


def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root
        (root / "media").mkdir(parents=True, exist_ok=True)

    def put(self, raw: bytes, mime: str) -> dict:
        digest = hashlib.sha256(raw).hexdigest()
        path = self.root / "media" / digest
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(raw)
            os.replace(tmp, path)
        return {"artifact": digest, "mime_type": mime, "bytes": len(raw)}

    def freeze(self, value):
        if isinstance(value, str) and value.startswith("data:image/"):
            header, data = value.split(",", 1)
            return self.put(base64.b64decode(data, validate=True), header[5:].split(";")[0])
        if isinstance(value, dict):
            return {
                k: "[REDACTED]"
                if k.lower() in ("authorization", "api_key", "password", "access_token")
                else self.freeze(v)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [self.freeze(x) for x in value]
        return value

    def thaw(self, value):
        if isinstance(value, dict) and "artifact" in value:
            digest = value["artifact"]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Invalid artifact digest")
            raw = (self.root / "media" / digest).read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("Artifact hash mismatch")
            return "data:" + value["mime_type"] + ";base64," + base64.b64encode(raw).decode()
        if isinstance(value, dict):
            return {k: self.thaw(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.thaw(v) for v in value]
        return value


class Journal:
    def __init__(self, root: Path):
        self.store = ArtifactStore(root)
        self.path = root / "requests.jsonl"
        self.sequence = 0
        if self.path.exists():
            raise FileExistsError("A session journal must never overwrite a prior attempt")

    def emit(self, kind: str, **payload):
        self.sequence += 1
        event = self.store.freeze(
            {"schema_version": 1, "seq": self.sequence, "time": time.time(), "kind": kind, **payload}
        )
        with self.path.open("a") as f:
            f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return event
