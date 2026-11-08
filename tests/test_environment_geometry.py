"""Distance probes must point along the camera/movement's left-turn convention."""

import math

import pytest

from navgpt.environment.geometry import bearing_direction
from navgpt.planner.tools.vla_client import _qmul, _qrot


@pytest.mark.parametrize("heading", [0, math.pi / 3, -2.2307788181626504, math.pi])
@pytest.mark.parametrize("bearing", [0, 40, -40, 90, -90, 180])
def test_walkable_direction_matches_rotated_camera(heading, bearing):
    rotation = (math.cos(heading / 2), 0, math.sin(heading / 2), 0)
    forward = _qrot(rotation, (0, 0, -1))
    angle = math.radians(bearing)
    turn = (math.cos(angle / 2), 0, math.sin(angle / 2), 0)
    expected = _qrot(_qmul(rotation, turn), (0, 0, -1))

    # The direction probe also normalizes projected forward vectors.
    actual = bearing_direction(3 * forward[0], 3 * forward[2], angle)
    assert actual == pytest.approx((expected[0], expected[2]), abs=1e-7)


def test_side_probes_face_the_labelled_side_of_an_asymmetric_room():
    def distance(direction):
        # Left wall is 2 m away; right wall is 7 m away at the identity pose.
        return 2 if direction[0] < 0 else 7

    left = bearing_direction(0, -1, math.pi / 2)
    right = bearing_direction(0, -1, -math.pi / 2)
    assert distance(left) == 2
    assert distance(right) == 7
