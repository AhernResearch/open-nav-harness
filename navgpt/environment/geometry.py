"""Habitat world directions, independent of simulator and array dependencies."""

import math


def bearing_direction(forward_x, forward_z, bearing_rad):
    """Unit XZ direction after a robot-relative bearing; positive turns left.

    Habitat looks along local -Z, so a +Y left turn points toward local -X.
    For a world forward vector (x, z), its left vector is therefore (z, -x).
    """
    norm = math.hypot(forward_x, forward_z) or 1.0
    fx, fz = forward_x / norm, forward_z / norm
    cosine, sine = math.cos(bearing_rad), math.sin(bearing_rad)
    return cosine * fx + sine * fz, cosine * fz - sine * fx
