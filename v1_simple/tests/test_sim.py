"""Sanity checks against hand calculations. Run:  .venv/bin/python -m pytest"""
import numpy as np
import pytest

from pathlib import Path

from simple_sim.motor import Motor, load_motor
from simple_sim.rocket import Rocket
from simple_sim.simulate import simulate
from simple_sim.physics import GRAVITY

MOTOR_FILE = Path(__file__).parents[2] / "data" / "AeroTech_HP-I500T.eng"


def test_motor_file_loads():
    m = load_motor(MOTOR_FILE)
    assert m.burn_time == pytest.approx(1.335)
    assert m.total_impulse == pytest.approx(620, rel=0.05)   # datasheet: 620 Ns
    assert m.mass(0) == pytest.approx(0.58)
    assert m.mass(m.burn_time) == pytest.approx(0.58 - 0.248)
    assert m.thrust(5.0) == 0.0


def test_matches_analytic_apogee():
    """Constant thrust for 1 s, no propellant mass loss, straight up -> closed-form answer."""
    T, m, burn = 100.0, 2.0, 1.0
    # Thrust drops to zero right after `burn`, like a real motor's final sample.
    motor = Motor("const", np.array([0, burn, burn + 1e-4]), np.array([T, T, 0.0]), propellant_mass=0.0, total_mass=0.0)
    rocket = Rocket(dry_mass=m, motor=motor, launch_angle_deg=0.0)

    a = T / m - GRAVITY
    v_burnout = a * burn
    h_burnout = 0.5 * a * burn**2
    expected_apogee = h_burnout + v_burnout**2 / (2 * GRAVITY)

    # Euler is first-order: error shrinks in proportion to dt, so use a small dt here.
    flight = simulate(rocket, dt=0.0001)
    assert flight.altitude.max() == pytest.approx(expected_apogee, rel=1e-3)


def test_too_heavy_stays_on_pad():
    motor = Motor("weak", np.array([0, 1.0]), np.array([5.0, 5.0]), 0.0, 0.0)
    flight = simulate(Rocket(dry_mass=2.0, motor=motor, launch_angle_deg=0.0), t_max=3)
    assert flight.altitude.max() == 0.0


def test_launch_angle_goes_downrange():
    motor = load_motor(MOTOR_FILE)
    flight = simulate(Rocket(1.5, motor, launch_angle_deg=10.0), dt=0.002)
    assert flight.position[-1, 0] > 100      # drifts in +x
    assert abs(flight.position[-1, 1]) < 1e-6


def test_controller_is_called_every_step():
    from simple_sim.controller import Controller
    from simple_sim.simulate import Simulation

    class Counter(Controller):
        calls = 0
        def update(self, t, dt, sensors):
            Counter.calls += 1
            return 3.0

    motor = load_motor(MOTOR_FILE)
    sim = Simulation(Rocket(1.5, motor, 0.0), Counter(), dt=0.01)
    for _ in range(10):
        sim.step()
    assert Counter.calls == 10
    assert sim.fin_deflection == 3.0
    assert sim.result().fin_deflection.tolist() == [3.0] * 10
