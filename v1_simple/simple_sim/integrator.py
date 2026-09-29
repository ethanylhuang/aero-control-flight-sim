"""Numerical integration. Kept separate so the method can be swapped (e.g. for RK4) later."""
import numpy as np


def step(position: np.ndarray, velocity: np.ndarray, accel: np.ndarray, dt: float):
    """Advance one time step. Returns (new_position, new_velocity).

    Semi-implicit Euler: update velocity first, then use the NEW velocity
    to update position. Same cost as plain Euler but more stable.
    """
    velocity = velocity + accel * dt      # integrate acceleration -> velocity
    position = position + velocity * dt   # integrate velocity -> position
    return position, velocity
