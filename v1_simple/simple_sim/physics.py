"""THE PHYSICS. Everything about forces lives in this one file.

Coordinates: x, y horizontal, z up. SI units.
When roll dynamics / aerodynamics are added later, they go here.
"""
import numpy as np

from .rocket import Rocket

GRAVITY = 9.81  # [m/s^2]


def net_force(t: float, position: np.ndarray, velocity: np.ndarray, rocket: Rocket) -> np.ndarray:
    """Total force on the rocket [N] as a 3-vector: thrust + gravity."""
    thrust = rocket.motor.thrust(t) * rocket.thrust_direction()
    gravity = rocket.mass(t) * np.array([0.0, 0.0, -GRAVITY])
    return thrust + gravity


def acceleration(t: float, position: np.ndarray, velocity: np.ndarray, rocket: Rocket) -> np.ndarray:
    """F = ma  ->  a = F / m   [m/s^2]"""
    return net_force(t, position, velocity, rocket) / rocket.mass(t)
