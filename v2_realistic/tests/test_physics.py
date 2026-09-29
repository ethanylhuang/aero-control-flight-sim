"""Sanity checks for the realistic sim. Run from the repo root:  .venv/bin/python -m pytest v2_realistic"""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from realistic_sim.atmosphere import Atmosphere, Wind, gravity
from realistic_sim.config import Table, load_rocket
from realistic_sim.controller import RollPIDController
from realistic_sim.integrator import launch_attitude, matrix_to_quat, quat_to_matrix
from realistic_sim.mass_properties import mass_properties
from realistic_sim.simulate import simulate

ROCKET_FILE = Path(__file__).parents[1] / "rockets" / "example" / "rocket.toml"


@pytest.fixture(scope="module")
def rocket():
    return load_rocket(ROCKET_FILE)


def with_settings(rk, **kw):
    return replace(rk, settings=replace(rk.settings, **kw))


def no_wind(rk):
    return replace(rk, wind=replace(rk.wind, speed_ms=0.0))


def test_standard_atmosphere():
    rho, a = Atmosphere().properties(0.0)
    assert rho == pytest.approx(1.225, abs=1e-3) and a == pytest.approx(340.3, abs=0.1)
    rho11, _ = Atmosphere().properties(11_000.0)
    assert rho11 == pytest.approx(0.3639, abs=2e-3)
    assert gravity(0.0) == pytest.approx(9.80665)


def test_wind_blows_toward_the_right_direction():
    east_wind_from_west = Wind(speed_ms=5, from_direction_deg=270, reference_height_m=10).vector(10.0)
    assert east_wind_from_west == pytest.approx([5, 0, 0], abs=1e-9)


def test_quaternion_roundtrip_and_launch_attitude():
    q = launch_attitude(20, 135)
    R = quat_to_matrix(q)
    assert np.allclose(R.T @ R, np.eye(3)) and np.allclose(matrix_to_quat(R), q)
    nose = R[:, 0]
    assert np.degrees(np.arccos(nose[2])) == pytest.approx(20)       # 20 degrees off vertical
    assert np.degrees(np.arctan2(nose[0], nose[1])) % 360 == pytest.approx(135)


def test_mass_properties_burn_down(rocket):
    args = (rocket.airframe, rocket.motor, rocket.motor_aft_position_m)
    start, end = mass_properties(*args, 0.0), mass_properties(*args, rocket.motor.burn_time)
    assert start.mass == pytest.approx(rocket.airframe.dry_mass_kg + rocket.motor.total_mass)
    assert start.mass - end.mass == pytest.approx(rocket.motor.propellant_mass)
    assert end.cg < start.cg          # the propellant is at the back, so CG moves forward as it burns


def test_vacuum_flight_matches_hand_calculation(rocket):
    """No air at all + constant thrust straight up: same closed form as the simple sim."""
    from realistic_sim.motor import Motor
    T, burn = 100.0, 1.0
    motor = Motor("const", np.array([0, burn, burn + 1e-4]), np.array([T, T, 0.0]),
                  propellant_mass=0.0, total_mass=1e-9, diameter_m=0.038, length_m=0.3)
    zero = Table(np.array([0.0, 1.0]), {c: np.zeros(2) for c in ("cd_coast", "cd_boost", "cn_alpha", "cp_m")})
    rk = replace(no_wind(rocket), motor=motor,
                 airframe=replace(rocket.airframe, dry_mass_kg=2.0),
                 aero=replace(rocket.aero, table=zero, roll_disturbance_coeff=0.0),
                 launch=replace(rocket.launch, angle_deg=0.0, rail_length_m=0.5,
                                site_elevation_m=0.0))
    m = 2.0 + 1e-9
    g = gravity(0.0)                      # gravity at altitude changes ~0.03% over 100 m: covered by tolerance
    a = T / m - g
    expected = 0.5 * a * burn**2 + (a * burn) ** 2 / (2 * g)
    flight = simulate(with_settings(rk, time_step_s=0.001))
    assert flight.altitude.max() == pytest.approx(expected, rel=2e-3)


def test_wind_makes_a_stable_rocket_weathercock_and_drift_upwind(rocket):
    rk = replace(rocket, launch=replace(rocket.launch, angle_deg=0.0),
                 wind=replace(rocket.wind, speed_ms=8.0, from_direction_deg=270.0))   # wind from the west
    flight = simulate(rk)
    assert flight.position[-1, 0] < -5.0             # ends up west = upwind of the pad
    # Right off the rail the rocket is slow, so a crosswind gives a big angle; once fast it must be small.
    assert np.degrees(flight.alpha[flight.airspeed > 60].max()) < 10


def test_unstable_rocket_tumbles(rocket):
    """Move the CP ahead of the CG: the rocket should not fly straight."""
    aero = replace(rocket.aero, table=Table(rocket.aero.table.mach, {
        **rocket.aero.table.columns, "cp_m": np.full_like(rocket.aero.table.mach, 0.6)}))
    flight = simulate(replace(rocket, aero=aero))
    assert np.degrees(flight.alpha[flight.airspeed > 20].max()) > 45


def test_roll_controller_beats_no_controller(rocket):
    free = simulate(rocket)
    held = simulate(rocket, RollPIDController())
    assert abs(free.roll_angle[-1]) > np.radians(90)              # the disturbance really spins it up
    assert np.abs(held.omega[:, 0]).max() < 0.2 * np.abs(free.omega[:, 0]).max()
    assert abs(held.roll_angle[-1]) < np.radians(10)


def test_flight_is_repeatable_and_time_step_converged(rocket):
    a = simulate(with_settings(rocket, time_step_s=0.001))
    b = simulate(with_settings(rocket, time_step_s=0.001))
    c = simulate(with_settings(rocket, time_step_s=0.0005, control_rate_hz=100.0))
    assert a.altitude.max() == b.altitude.max()                   # same random seed -> identical
    assert a.altitude.max() == pytest.approx(c.altitude.max(), rel=2e-3)
