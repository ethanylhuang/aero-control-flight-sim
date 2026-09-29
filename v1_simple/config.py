"""All the settings you'd want to tweak. Used by run_sim.py and live_view.py."""
from pathlib import Path

from simple_sim.motor import load_motor
from simple_sim.rocket import Rocket
from simple_sim.controller import NoController

HERE = Path(__file__).parent
MOTOR_FILE = HERE.parent / "data" / "AeroTech_HP-I500T.eng"   # shared thrust curve
DRY_MASS = 1.5            # kg, rocket without motor  (PLACEHOLDER: use the real number)
LAUNCH_ANGLE_DEG = 5.0    # tilt from vertical (0 = straight up)
LAUNCH_HEADING_DEG = 0.0  # direction of the tilt in the x-y plane
TIME_STEP = 0.001         # s

CONTROLLER = NoController()   # <- swap in the roll controller here


def make_rocket() -> Rocket:
    motor = load_motor(MOTOR_FILE)
    return Rocket(DRY_MASS, motor, LAUNCH_ANGLE_DEG, LAUNCH_HEADING_DEG)
