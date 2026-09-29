"""Rocket description: the inputs to the simulation."""
from dataclasses import dataclass
import numpy as np

from .motor import Motor


@dataclass
class Rocket:
    dry_mass: float           # rocket mass WITHOUT the motor [kg]
    motor: Motor
    launch_angle_deg: float   # tilt from vertical: 0 = straight up
    launch_heading_deg: float = 0.0  # compass-style direction of the tilt in the x-y plane

    def mass(self, t: float) -> float:
        """Total mass [kg] at time t (shrinks as propellant burns)."""
        return self.dry_mass + self.motor.mass(t)

    def thrust_direction(self) -> np.ndarray:
        """Unit vector the motor pushes along. Fixed for now (no aerodynamic turning)."""
        tilt = np.radians(self.launch_angle_deg)
        heading = np.radians(self.launch_heading_deg)
        return np.array([
            np.sin(tilt) * np.cos(heading),   # x
            np.sin(tilt) * np.sin(heading),   # y
            np.cos(tilt),                     # z (up)
        ])
