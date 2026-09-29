"""Fin servo: the fins can't jump to the command instantly, and can't exceed their travel."""
import numpy as np


class Actuator:
    def __init__(self, max_deflection_deg: float, max_rate_deg_s: float):
        self.max_deflection = max_deflection_deg
        self.max_rate = max_rate_deg_s
        self.position = 0.0

    def update(self, command_deg: float, dt: float) -> float:
        target = np.clip(command_deg, -self.max_deflection, self.max_deflection)
        step = np.clip(target - self.position, -self.max_rate * dt, self.max_rate * dt)
        self.position += step
        return self.position
