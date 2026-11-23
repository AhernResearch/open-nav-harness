"""Observed map: unprojection into a fixed world frame, the height band, growth and cropping."""

import io

import numpy as np
from PIL import Image

from open_nav_harness.mapping.observed import DepthView, MapSpec, ObservedMap

# A camera 1 m above the floor at the origin, looking straight down (-Z camera axis = -Y world).
DOWN = np.array([-np.sqrt(0.5), 0, 0, np.sqrt(0.5)])  # -90 deg about +X
K = np.array([[20.0, 0, 15.5], [0, 20.0, 15.5], [0, 0, 1]])


def view(depth, colour=(200, 40, 40), position=(0.0, 1.0, 0.0), rotation=DOWN):
    rgb = np.zeros((32, 32, 3), np.uint8)
    rgb[:] = colour
    return DepthView(rgb, np.full((32, 32), depth, np.float32), K, np.array(position),
                     np.array(rotation))


def pixel(image, frame, x, z):
    m = frame["matrix"]
    return image[int(m[3] * x + m[4] * z + m[5]), int(m[0] * x + m[1] * z + m[2])]


def decode(png):
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def test_floor_seen_from_above_lands_under_the_camera_in_its_colour():
    observed = ObservedMap(MapSpec(pixel_stride=1))
    observed.add(view(1.0), floor_y=0.0)
    png, frame = observed.render([[0.0, 0.0, 0.0]])
    image = decode(png)
    red = pixel(image, frame, 0.0, 0.0)
    assert red[0] > 150 and red[1] < 100
    # 32 px at f=20 over 1 m covers 1.6 m; 3 m away nothing was measured.
    assert tuple(pixel(image, frame, 2.5, 0.0)) in {(46, 52, 63), (58, 66, 79)}
    assert frame["seen_area_m2"] > 2.0


def test_points_outside_the_height_band_are_not_painted():
    observed = ObservedMap(MapSpec(pixel_stride=1))
    observed.add(view(0.2), floor_y=0.0)  # a surface at 0.8 m: kept
    observed.add(view(3.0, colour=(10, 200, 10)), floor_y=0.0)  # 2 m below the floor: dropped
    _, frame = observed.render([[0.0, 0.0, 0.0]])
    assert observed.count.sum() > 0 and frame["seen_area_m2"] < 0.2


def test_the_grid_grows_without_moving_earlier_cells_and_the_crop_keeps_the_route():
    observed = ObservedMap(MapSpec(pixel_stride=1))
    observed.add(view(1.0), floor_y=0.0)
    before = observed.count.sum()
    observed.add(view(1.0, colour=(40, 40, 200), position=(30.0, 1.0, -20.0)), floor_y=0.0)
    png, frame = observed.render([[0.0, 0.0, 0.0], [30.0, 0.0, -20.0]])
    image = decode(png)
    assert observed.count.sum() == 2 * before
    assert pixel(image, frame, 0.0, 0.0)[0] > 150 and pixel(image, frame, 30.0, -20.0)[2] > 150
    assert frame["size"][0] >= 30 * frame["ppm"]


def test_a_first_render_with_no_coverage_still_has_the_minimum_span():
    png, frame = ObservedMap(MapSpec()).render([[1.0, 0.0, 2.0]])
    assert min(frame["size"]) >= 6.0 * frame["ppm"] - 1
    assert decode(png).shape[:2] == (frame["size"][1], frame["size"][0])
