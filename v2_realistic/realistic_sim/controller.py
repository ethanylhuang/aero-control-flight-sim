"""Controller plug-in point.

Each control tick (config: control_rate_hz) the simulation hands the controller
SENSOR READINGS (noisy, like a real flight computer would get) and it returns a
fin deflection command in degrees. The command is then limited by the actuator's
max deflection and max rate before it reaches the fins.

To plug in your own: subclass Controller, then pass an instance to Simulation(...).
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class SensorReadings:
    t: float                    # s
    gyro: np.ndarray            # body angular rates [rad/s]: (roll p, pitch q, yaw r)
    accel: np.ndarray           # accelerometer, body frame [m/s^2] (specific force: reads +g when sitting still)
    baro_altitude: float        # m above the launch site


class Controller:
    def update(self, t: float, dt: float, readings: SensorReadings) -> float:
        """Return the commanded fin deflection [deg]."""
        raise NotImplementedError


class NoController(Controller):
    """Fins stay centered."""
    def update(self, t, dt, readings):
        return 0.0


class RollPIDController(Controller):
    """EXAMPLE roll-angle PID, only here to demonstrate the plumbing. Gains are untuned guesses.

    Holds roll at target_deg. Its roll angle is its own integral of the gyro, as on a real rocket.
    """
    def __init__(self, kp=60.0, ki=10.0, kd=8.0, target_deg=0.0, integral_limit=0.3):
        self.kp, self.ki, self.kd = kp, ki, kd     # units: deg per rad, deg per (rad s), deg per (rad/s)
        self.target = np.radians(target_deg)
        self.integral_limit = integral_limit
        self.angle = 0.0
        self.integral = 0.0

    def update(self, t, dt, readings):
        p = readings.gyro[0]
        self.angle += p * dt
        error = self.target - self.angle
        self.integral = float(np.clip(self.integral + error * dt, -self.integral_limit, self.integral_limit))
        return self.kp * error + self.ki * self.integral - self.kd * p
