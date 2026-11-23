"""What the robot has actually seen, painted onto a top-down colour grid.

Every observed RGB-D view is unprojected into world space; points within a height band
over the current floor are averaged per cell. Only cells a camera measured are coloured,
so rooms never looked into stay unknown. The grid grows with coverage and keeps one fixed
world frame, so successive renders are directly comparable. North stays up; markers are
not burnt in, a renderer draws them from the returned world-to-pixel transform.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image
from pydantic import Field

from open_nav_harness.core.config import StrictModel

UNKNOWN = ((46, 52, 63), (58, 66, 79))  # faint checker: never as dark as a measured cell
MEASURED_FLOOR = 88.0
GROW_M = 8.0
FLOOR_BAND_M = 0.15  # samples this close to the floor height get the grazing-angle stretch


class MapSpec(StrictModel):
    """NavGPT-3's observed_map parameters, with a finer stride and splat cap for the low camera."""

    enabled: bool = True
    mpp: float = Field(default=0.02, gt=0.005, le=0.2)
    depth_max_m: float = Field(default=10.0, gt=0, le=50)
    pixel_stride: int = Field(default=2, ge=1, le=16)
    below_floor_m: float = Field(default=0.30, ge=0, le=2)
    ceiling_m: float = Field(default=1.60, gt=0, le=5)
    splat_max_m: float = Field(default=0.16, ge=0, le=1)
    margin_m: float = Field(default=1.5, ge=0, le=10)
    min_span_m: float = Field(default=6.0, gt=0, le=100)


@dataclass(frozen=True)
class DepthView:
    """One camera: RGB, axial depth in metres, pinhole K and the world sensor pose."""

    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    position: np.ndarray
    rotation_xyzw: np.ndarray


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate row vectors ``v`` by the xyzw quaternion ``q``."""
    u, w = q[:3], q[3]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


class ObservedMap:
    def __init__(self, spec: MapSpec):
        self.spec = spec
        self.lo: np.ndarray | None = None  # world (x, z) of cell (0, 0)
        self.sum: np.ndarray = np.zeros((0, 0, 3), np.float64)
        self.count: np.ndarray = np.zeros((0, 0), np.int64)

    def cells(self, xz: np.ndarray) -> np.ndarray:
        assert self.lo is not None
        return np.rint((xz - self.lo) / self.spec.mpp).astype(np.int64)  # (col, row)

    def ensure(self, xz: np.ndarray):
        """Grow the grid by whole blocks so it covers ``xz``; existing cells keep their place."""
        mpp, pad = self.spec.mpp, GROW_M
        if self.lo is None:
            self.lo = np.floor((xz.min(0) - pad) / mpp) * mpp
            hi = xz.max(0) + pad
            shape = np.ceil((hi - self.lo) / mpp).astype(int) + 1
            self.sum = np.zeros((shape[1], shape[0], 3), np.float64)
            self.count = np.zeros((shape[1], shape[0]), np.int64)
            return
        low, high = xz.min(0), xz.max(0)
        width, height = self.count.shape[1], self.count.shape[0]
        top = self.lo + np.array([width - 1, height - 1]) * mpp
        if (low >= self.lo).all() and (high <= top).all():
            return
        lo = np.where(low < self.lo, np.floor((low - pad) / mpp) * mpp, self.lo)
        hi = np.maximum(top, np.where(high > top, high + pad, top))
        shape = np.ceil((hi - lo) / mpp).astype(int) + 1
        shift = np.rint((self.lo - lo) / mpp).astype(int)
        grown_sum = np.zeros((shape[1], shape[0], 3), np.float64)
        grown_count = np.zeros((shape[1], shape[0]), np.int64)
        grown_sum[shift[1]:shift[1] + height, shift[0]:shift[0] + width] = self.sum
        grown_count[shift[1]:shift[1] + height, shift[0]:shift[0] + width] = self.count
        self.lo, self.sum, self.count = lo, grown_sum, grown_count

    def add(self, view: DepthView, floor_y: float):
        """Accumulate one view; samples are area-splatted so the far field has no spokes."""
        spec, stride = self.spec, self.spec.pixel_stride
        depth = view.depth[::stride, ::stride]
        rgb = view.rgb[::stride, ::stride, :3]
        k = view.intrinsics
        v, u = np.mgrid[0:view.depth.shape[0]:stride, 0:view.depth.shape[1]:stride]
        good = np.isfinite(depth) & (depth > 1e-3) & (depth < spec.depth_max_m)
        d = depth[good]
        local = np.stack([(u[good] - k[0, 2]) * d / k[0, 0], -(v[good] - k[1, 2]) * d / k[1, 1], -d], -1)
        world = rotate(view.rotation_xyzw, local) + view.position
        band = (world[:, 1] > floor_y - spec.below_floor_m) & (world[:, 1] < floor_y + spec.ceiling_m)
        world, colours = world[band], rgb[good][band].astype(np.float64)
        if not len(world):
            return
        xz = world[:, [0, 2]]
        self.ensure(xz)
        cell = self.cells(xz)
        # A sample at range r spans about r * dtheta across the ray (NavGPT-3's splat). The one
        # departure from NavGPT-3: our camera sits at 0.88 m rather than 1.25 m, so far floor is
        # seen at a grazing angle and adjacent rows land up to ~0.3 m apart along the ray, which
        # the across-ray splat leaves as stripes. Floor samples therefore also stretch by
        # 1/tan(elevation); walls and furniture keep NavGPT-3's splat.
        dtheta = 2 * np.arctan(view.depth.shape[1] / (2 * k[0, 0])) / (view.depth.shape[1] / stride)
        reach = int(max(1, round(spec.splat_max_m / spec.mpp)))
        ray = world - view.position
        distance = np.linalg.norm(ray, axis=1)
        rise = np.abs(ray[:, 1]) / np.maximum(distance, 1e-6)
        floor = np.abs(world[:, 1] - floor_y) < FLOOR_BAND_M
        stretch = np.where(floor, np.maximum(1.0, np.sqrt(1 - rise ** 2) / np.maximum(rise, 1e-3)), 1.0)
        radius = np.clip(np.rint(distance * dtheta * stretch / spec.mpp / 2), 0, reach).astype(int)
        height, width = self.count.shape
        offsets = [(dr, dc) for dr in range(-reach, reach + 1) for dc in range(-reach, reach + 1)]
        picks = [np.nonzero(radius >= max(abs(dr), abs(dc)))[0] for dr, dc in offsets]
        col = np.concatenate([cell[p, 0] + dc for p, (_, dc) in zip(picks, offsets)])
        row = np.concatenate([cell[p, 1] + dr for p, (dr, _) in zip(picks, offsets)])
        source = np.concatenate(picks)
        inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
        index, source = row[inside] * width + col[inside], source[inside]
        # Only the touched bounding range is counted, not the whole grid.
        low = index.min()
        index -= low
        size = index.max() + 1
        flat_sum, flat_count = self.sum.reshape(-1, 3), self.count.reshape(-1)
        for channel in range(3):
            flat_sum[low:low + size, channel] += np.bincount(index, colours[source, channel], size)
        flat_count[low:low + size] += np.bincount(index, minlength=size)

    def render(self, route: list[list[float]]) -> tuple[bytes, dict]:
        """PNG cropped to the route and its observed surroundings, plus its world-to-pixel frame."""
        spec = self.spec
        route_xz = np.asarray([[p[0], p[2]] for p in route], float)
        self.ensure(route_xz)
        assert self.lo is not None
        height, width = self.count.shape
        x0, y0 = self.cells(route_xz).min(0)
        x1, y1 = self.cells(route_xz).max(0)
        rows, cols = np.nonzero(self.count)
        if rows.size:  # percentiles, so sparse sight-line whiskers do not set the extent
            y0, y1 = min(y0, int(np.percentile(rows, 2))), max(y1, int(np.percentile(rows, 98)))
            x0, x1 = min(x0, int(np.percentile(cols, 2))), max(x1, int(np.percentile(cols, 98)))
        margin, span = round(spec.margin_m / spec.mpp), round(spec.min_span_m / spec.mpp)
        x0, x1 = self._span(x0 - margin, x1 + margin, span, width)
        y0, y1 = self._span(y0 - margin, y1 + margin, span, height)
        count, total = self.count[y0:y1 + 1, x0:x1 + 1], self.sum[y0:y1 + 1, x0:x1 + 1]
        seen = count > 0
        image = np.empty((*count.shape, 3), np.uint8)
        yy, xx = np.mgrid[0:count.shape[0], 0:count.shape[1]]
        image[:] = UNKNOWN[0]
        image[((yy // 4) + (xx // 4)) % 2 == 0] = UNKNOWN[1]
        if seen.any():
            mean = total / np.maximum(count, 1)[..., None]
            # Gamma lifts shadows without washing hue out; a floor keeps dark surfaces
            # distinguishable from never-observed cells.
            mean = np.maximum(255 * (np.clip(mean, 0, 255) / 255) ** 0.85, MEASURED_FLOOR)
            image[seen] = mean[seen].astype(np.uint8)
            fill, colours = self._holes(seen, mean)
            image[fill] = colours
        stream = io.BytesIO()
        Image.fromarray(image).save(stream, format="PNG")
        ppm = 1 / spec.mpp
        origin = self.lo + np.array([x0, y0]) * spec.mpp
        # px = m0*x + m1*z + m2 ; py = m3*x + m4*z + m5, with cell centres at pixel centres.
        matrix = [ppm, 0.0, 0.5 - origin[0] * ppm, 0.0, ppm, 0.5 - origin[1] * ppm]
        return stream.getvalue(), {"size": [image.shape[1], image.shape[0]], "ppm": ppm, "matrix": matrix,
                                   "seen_area_m2": float((self.count > 0).sum()) * spec.mpp ** 2}

    @staticmethod
    def _holes(seen: np.ndarray, mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Single-cell gaps between splats: unseen cells with 5+ seen neighbours, never the border."""
        colour = np.where(seen[..., None], mean, 0.0)
        acc_colour, acc_seen = np.zeros_like(colour), np.zeros(seen.shape)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    acc_colour += np.roll(colour, (dy, dx), (0, 1))
                    acc_seen += np.roll(seen, (dy, dx), (0, 1))
        fill = ~seen & (acc_seen >= 5)
        fill[[0, -1], :] = fill[:, [0, -1]] = False
        return fill, (acc_colour[fill] / acc_seen[fill][:, None]).astype(np.uint8)

    @staticmethod
    def _span(low: int, high: int, span: int, size: int) -> tuple[int, int]:
        if high - low + 1 < span:
            centre = (low + high) // 2
            low, high = centre - span // 2, centre - span // 2 + span - 1
        return max(0, low), min(size - 1, high)
