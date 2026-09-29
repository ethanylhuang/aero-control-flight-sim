"""Motor: loads a RASP .eng file (thrustcurve.org). Data handling only, no flight physics."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Motor:
    name: str
    times: np.ndarray        # s, starts at 0
    thrusts: np.ndarray      # N
    propellant_mass: float   # kg
    total_mass: float        # kg, before ignition
    diameter_m: float
    length_m: float

    def __post_init__(self):
        avg = (self.thrusts[1:] + self.thrusts[:-1]) / 2
        self._cum_impulse = np.concatenate([[0.0], np.cumsum(avg * np.diff(self.times))])

    @property
    def burn_time(self) -> float:
        return float(self.times[-1])

    @property
    def total_impulse(self) -> float:
        return float(self._cum_impulse[-1])

    def thrust(self, t: float) -> float:
        return float(np.interp(t, self.times, self.thrusts, left=0.0, right=0.0))

    def mass(self, t: float) -> float:
        """Motor mass at time t: propellant burns off in proportion to impulse delivered."""
        burned = np.interp(t, self.times, self._cum_impulse) / self.total_impulse
        return self.total_mass - self.propellant_mass * burned


def load_motor(path) -> Motor:
    """Header: <name> <diameter mm> <length mm> <delays> <propellant kg> <total kg> <maker>, then <time> <thrust> rows."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith(";")]
    h = lines[0].split()
    data = np.array([[float(x) for x in ln.split()] for ln in lines[1:]])
    times = np.concatenate([[0.0], data[:, 0]])       # RASP omits the t=0 point (thrust is 0 at ignition)
    thrusts = np.concatenate([[0.0], data[:, 1]])
    return Motor(h[0], times, thrusts, float(h[4]), float(h[5]), float(h[1]) / 1000, float(h[2]) / 1000)
