"""The simulation loop. Ties together dynamics, integrator, sensors, controller and actuator.

    flight = simulate(rocket)                 -> run to the end, get a FlightResult
    sim = Simulation(rocket, controller)      -> call sim.step() yourself (live viewer)

Each physics step (time_step_s):
    1. if it's a control tick: read sensors -> controller -> new fin command
    2. the fins move toward the command (rate/travel limited)
    3. RK4 advances the 6-DOF state
The controller runs at control_rate_hz (e.g. 100 Hz) while physics runs faster (e.g. 1000 Hz).
"""
from dataclasses import dataclass
import numpy as np

from .actuator import Actuator
from .atmosphere import Atmosphere, Wind
from .config import RocketConfig
from .controller import Controller, NoController
from .dynamics import Dynamics, Environment, POS, VEL, QUAT, OMEGA, ROLL
from .integrator import rk4_step, launch_attitude
from .sensors import SensorSuite

LOG_FIELDS = ("t", "position", "velocity", "quat", "omega", "roll_angle", "mach", "q", "alpha",
              "airspeed", "mass", "cg", "thrust", "accel", "command", "deflection")


@dataclass
class FlightResult:
    t: np.ndarray
    position: np.ndarray     # (N, 3) world frame [m]
    velocity: np.ndarray     # (N, 3) world frame [m/s]
    quat: np.ndarray         # (N, 4) attitude
    omega: np.ndarray        # (N, 3) body rates [rad/s]: roll, pitch, yaw
    roll_angle: np.ndarray   # [rad]
    mach: np.ndarray
    q: np.ndarray            # dynamic pressure [Pa]
    alpha: np.ndarray        # angle of attack [rad]
    airspeed: np.ndarray     # [m/s] relative to the air
    mass: np.ndarray
    cg: np.ndarray           # [m from nose]
    thrust: np.ndarray
    accel: np.ndarray        # (N, 3) accelerometer (body frame, specific force) [m/s^2]
    command: np.ndarray      # controller command [deg]
    deflection: np.ndarray   # actual fin deflection [deg]
    rail_exit_speed: float = float("nan")

    @property
    def altitude(self):
        return self.position[:, 2]

    @property
    def speed(self):
        return np.linalg.norm(self.velocity, axis=1)


class Simulation:
    def __init__(self, rocket: RocketConfig, controller: Controller = None):
        self.rocket = rocket
        self.controller = controller or NoController()
        st = rocket.settings
        self.dt = st.time_step_s
        self.control_every = max(1, round(1.0 / (st.control_rate_hz * self.dt)))
        self.control_dt = self.control_every * self.dt

        env = Environment(
            Atmosphere(rocket.launch.site_elevation_m, rocket.launch.temperature_offset_k),
            Wind(rocket.wind.speed_ms, rocket.wind.from_direction_deg,
                 rocket.wind.reference_height_m, rocket.wind.shear_exponent),
        )
        self.dynamics = Dynamics(rocket, env)
        self.sensors = SensorSuite(rocket.sensors, st.random_seed)
        self.actuator = Actuator(rocket.control_fins.max_deflection_deg, rocket.control_fins.max_rate_deg_s)

        s = np.zeros(14)
        s[QUAT] = launch_attitude(rocket.launch.angle_deg, rocket.launch.heading_deg)
        self.state = s
        self.steps = 0
        self.t = 0.0
        self.command = 0.0
        self.on_rail = True
        self.done = False
        self.rail_exit_speed = float("nan")
        self._info = self.dynamics.derivatives(0.0, s, 0.0, True)[1]
        self._log = {k: [] for k in LOG_FIELDS}

    # convenient views of the current state
    position = property(lambda self: self.state[POS])
    velocity = property(lambda self: self.state[VEL])
    omega = property(lambda self: self.state[OMEGA])
    roll_angle = property(lambda self: self.state[ROLL])
    deflection = property(lambda self: self.actuator.position)

    def step(self):
        if self.done:
            return
        s, t, dt = self.state, self.t, self.dt

        if self.steps % self.control_every == 0:
            readings = self.sensors.read(t, s[OMEGA], self._info["specific_force"], s[2])
            self.command = float(self.controller.update(t, self.control_dt, readings))
        self.actuator.update(self.command, dt)

        f = lambda tt, ss: self.dynamics.derivatives(tt, ss, self.actuator.position, self.on_rail)
        new_state, self._info = rk4_step(f, t, s, dt)
        self._record(t, s)
        self.state = new_state
        self.steps += 1
        self.t = self.steps * dt

        pos = self.state[POS]
        if self.on_rail and np.linalg.norm(pos) >= self.rocket.launch.rail_length_m:
            self.on_rail = False
            self.rail_exit_speed = float(np.linalg.norm(self.state[VEL]))
        if not self.on_rail:
            landed = pos[2] < 0.0
            apogee = self.rocket.settings.stop_at_apogee and self.state[5] < 0.0
            self.done = landed or apogee
        if self.t >= self.rocket.settings.max_time_s:
            self.done = True

    def run(self) -> FlightResult:
        while not self.done:
            self.step()
        return self.result()

    def result(self) -> FlightResult:
        arrays = {k: np.array(v) for k, v in self._log.items()}
        return FlightResult(rail_exit_speed=self.rail_exit_speed, **arrays)

    def _record(self, t, s):
        i, log = self._info, self._log
        aero = i["aero"]
        log["t"].append(t)
        log["position"].append(s[POS])
        log["velocity"].append(s[VEL])
        log["quat"].append(s[QUAT])
        log["omega"].append(s[OMEGA])
        log["roll_angle"].append(s[ROLL])
        log["mach"].append(aero.mach)
        log["q"].append(aero.q)
        log["alpha"].append(aero.alpha)
        log["airspeed"].append(aero.airspeed)
        log["mass"].append(i["mass"])
        log["cg"].append(i["cg"])
        log["thrust"].append(i["thrust"])
        log["accel"].append(i["specific_force"])
        log["command"].append(self.command)
        log["deflection"].append(self.actuator.position)


def simulate(rocket: RocketConfig, controller: Controller = None) -> FlightResult:
    return Simulation(rocket, controller).run()
