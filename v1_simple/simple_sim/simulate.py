"""The simulation loop: ties physics + integrator + controller together and records the flight.

Two ways to use it:
    simulate(rocket)            -> run the whole flight, get a FlightResult
    sim = Simulation(rocket)    -> call sim.step() yourself (used by the live viewer)
"""
from dataclasses import dataclass
import numpy as np

from .rocket import Rocket
from .physics import acceleration
from .integrator import step
from .controller import Controller, NoController, Sensors


@dataclass
class FlightResult:
    t: np.ndarray               # [s]
    position: np.ndarray        # shape (N, 3) [m]   columns: x, y, z(altitude)
    velocity: np.ndarray        # shape (N, 3) [m/s]
    acceleration: np.ndarray    # shape (N, 3) [m/s^2]
    mass: np.ndarray            # [kg]
    thrust: np.ndarray          # [N]
    fin_deflection: np.ndarray  # controller output [deg]

    @property
    def altitude(self) -> np.ndarray:
        return self.position[:, 2]

    @property
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.velocity, axis=1)


class Simulation:
    def __init__(self, rocket: Rocket, controller: Controller = None, dt: float = 0.001, t_max: float = 300.0):
        self.rocket = rocket
        self.controller = controller or NoController()
        self.dt = dt
        self.t_max = t_max

        # Current state
        self.steps = 0          # time is steps * dt (avoids float drift from repeated t += dt)
        self.t = 0.0
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.fin_deflection = 0.0
        self.launched = False
        self.done = False

        self._log = {k: [] for k in FlightResult.__dataclass_fields__}

    def step(self):
        """Advance the simulation by one time step."""
        if self.done:
            return

        accel = acceleration(self.t, self.position, self.velocity, self.rocket)

        # On the ground and being pushed down (or not up): the pad holds it still.
        if self.position[2] <= 0 and accel[2] <= 0:
            if self.launched:
                self.done = True    # already flew -> this is the landing
                return
            accel = np.zeros(3)     # still waiting on the pad
            self.velocity = np.zeros(3)
        self.acceleration = accel

        sensors = Sensors(self.position, self.velocity, accel)
        self.fin_deflection = self.controller.update(self.t, self.dt, sensors)

        self._record()
        self.position, self.velocity = step(self.position, self.velocity, accel, self.dt)
        self.steps += 1
        self.t = self.steps * self.dt

        if self.position[2] > 0:
            self.launched = True
        if self.t >= self.t_max:
            self.done = True

    def run(self) -> FlightResult:
        """Step until landing, then return the full recorded flight."""
        while not self.done:
            self.step()
        return self.result()

    def result(self) -> FlightResult:
        return FlightResult(**{k: np.array(v) for k, v in self._log.items()})

    def _record(self):
        self._log["t"].append(self.t)
        self._log["position"].append(self.position)
        self._log["velocity"].append(self.velocity)
        self._log["acceleration"].append(self.acceleration)
        self._log["mass"].append(self.rocket.mass(self.t))
        self._log["thrust"].append(self.rocket.motor.thrust(self.t))
        self._log["fin_deflection"].append(self.fin_deflection)


def simulate(rocket: Rocket, controller: Controller = None, dt: float = 0.001, t_max: float = 300.0) -> FlightResult:
    """Fly the rocket from the pad until it lands (or t_max)."""
    return Simulation(rocket, controller, dt, t_max).run()
