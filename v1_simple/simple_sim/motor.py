"""Rocket motor: loads a RASP .eng file and answers "how much thrust / how heavy at time t?"

This is data-handling only, no flight physics lives here.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Motor:
    name: str
    times: np.ndarray        # thrust curve sample times [s], starts at 0
    thrusts: np.ndarray      # thrust at those times [N]
    propellant_mass: float   # [kg]
    total_mass: float        # motor mass before ignition, propellant + casing [kg]

    def __post_init__(self):
        # Cumulative impulse at each sample (trapezoid rule), used to work out
        # how much propellant has burned by time t.
        dt = np.diff(self.times)
        avg_thrust = (self.thrusts[1:] + self.thrusts[:-1]) / 2
        self._cum_impulse = np.concatenate([[0.0], np.cumsum(avg_thrust * dt)])

    @property
    def burn_time(self) -> float:
        return float(self.times[-1])

    @property
    def total_impulse(self) -> float:
        return float(self._cum_impulse[-1])

    def thrust(self, t: float) -> float:
        """Thrust [N] at time t. Zero before ignition and after burnout."""
        return float(np.interp(t, self.times, self.thrusts, left=0.0, right=0.0))

    def mass(self, t: float) -> float:
        """Motor mass [kg] at time t. Propellant burns off in proportion to impulse delivered."""
        burned_impulse = np.interp(t, self.times, self._cum_impulse)
        burned_fraction = burned_impulse / self.total_impulse
        return self.total_mass - self.propellant_mass * burned_fraction


def load_motor(path: str) -> Motor:
    """Read the first motor in a RASP .eng file (as downloaded from thrustcurve.org).

    File layout:
        ; comment lines
        <name> <diameter mm> <length mm> <delays> <propellant kg> <total kg> <maker>
        <time s> <thrust N>
        ...
    """
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith(";")]

    header = lines[0].split()
    name = header[0]
    propellant_mass = float(header[4])
    total_mass = float(header[5])

    data = np.array([[float(x) for x in ln.split()] for ln in lines[1:]])
    times, thrusts = data[:, 0], data[:, 1]

    # RASP curves omit the t=0 point; thrust is zero at ignition.
    times = np.concatenate([[0.0], times])
    thrusts = np.concatenate([[0.0], thrusts])

    return Motor(name, times, thrusts, propellant_mass, total_mass)
