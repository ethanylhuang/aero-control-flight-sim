"""Controller plug-in point.

A controller is anything with an update() method. Each time step the simulation
hands it the sensor readings and it returns the fin deflection command [deg].

To plug in the roll controller: subclass Controller (e.g. a PIDController),
then set CONTROLLER in config.py. Nothing else needs to change.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Sensors:
    """What the controller is allowed to "see". Add roll_angle / roll_rate here later."""
    position: np.ndarray       # [m]
    velocity: np.ndarray       # [m/s]
    acceleration: np.ndarray   # [m/s^2]


class Controller:
    def update(self, t: float, dt: float, sensors: Sensors) -> float:
        """Return the commanded fin deflection [deg]."""
        raise NotImplementedError


class NoController(Controller):
    """Fins stay centered. Default until the real controller exists."""
    def update(self, t, dt, sensors):
        return 0.0
