"""Read-only HTTP responses for recorded media: verified artifacts and image files."""

import hashlib
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def artifact_response(media: Path, digest: str) -> Response:
    """A content-addressed file from ``media``, re-verified before it is served."""
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise HTTPException(404)
    path = media / digest
    if not path.is_file():
        raise HTTPException(404)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise HTTPException(409, "Artifact digest mismatch")
    kind = ("image/png" if raw.startswith(b"\x89PNG") else
            "image/webp" if raw.startswith(b"RIFF") else "image/jpeg")
    # Content-addressed and verified above, so it can never change under this URL.
    return Response(raw, media_type=kind, headers={"Cache-Control": "private, max-age=31536000, immutable"})


def image_file(directory: Path, relative: str) -> FileResponse:
    path = (directory / relative).resolve()
    if (not path.is_relative_to(directory.resolve()) or path.suffix.lower() not in IMAGE_SUFFIXES
            or not path.is_file()):
        raise HTTPException(404)
    return FileResponse(path)
