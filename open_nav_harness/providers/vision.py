"""Image token estimate from resolution, for gateways that expose no token counter.

The arithmetic is the Qwen-VL processor's ``smart_resize``: the image is snapped to a grid of
32-pixel cells (16-pixel patches merged 2x2), scaled into the processor's pixel bounds, and
each cell is one token, plus the vision start and end markers. Checked against the 27B vLLM
``/tokenize`` from 128x96 to 4000x3000, where it matches exactly. Other gateways tokenize
images their own way; this is a resolution-aware budget, not their exact count.
"""

from __future__ import annotations

import base64
import binascii
import io
import math

from PIL import Image, UnidentifiedImageError

CELL = 32
MIN_PIXELS = 64 * CELL * CELL
MAX_PIXELS = 16384 * CELL * CELL
MARKERS = 2


def vision_tokens(width: int, height: int) -> int:
    rows, cols = max(1, round(height / CELL)), max(1, round(width / CELL))
    if rows * cols * CELL * CELL > MAX_PIXELS:
        scale = math.sqrt(width * height / MAX_PIXELS)
        rows, cols = math.floor(height / scale / CELL), math.floor(width / scale / CELL)
    elif rows * cols * CELL * CELL < MIN_PIXELS:
        scale = math.sqrt(MIN_PIXELS / (width * height))
        rows, cols = math.ceil(height * scale / CELL), math.ceil(width * scale / CELL)
    return rows * cols + MARKERS


def image_size(url: str) -> tuple[int, int]:
    """Width and height of an inline ``data:`` image, read from its header only."""
    header, _, payload = url.partition(",")
    if not header.startswith("data:") or not header.endswith(";base64"):
        raise ValueError("token estimate needs inline base64 images")
    try:
        with Image.open(io.BytesIO(base64.b64decode(payload, validate=True))) as image:
            return image.size
    except (binascii.Error, UnidentifiedImageError) as error:
        raise ValueError("token estimate could not read an inline image") from error
