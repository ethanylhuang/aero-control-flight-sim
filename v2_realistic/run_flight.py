"""Fly the rocket with and without the roll controller, print OpenRocket-style numbers, plot both.

    .venv/bin/python v2_realistic/run_flight.py                       # example rocket
    .venv/bin/python v2_realistic/run_flight.py --rocket path/to/rocket.toml
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from realistic_sim.analysis import summarize
from realistic_sim.config import load_rocket
from realistic_sim.controller import NoController, RollPIDController
from realistic_sim.simulate import simulate

HERE = Path(__file__).parent

# ---------------- which controllers to compare (swap in your own here) ----------------
CONTROLLERS = {
    "no control": NoController,
    "example PID": RollPIDController,
}
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--rocket", default=HERE / "rockets" / "example" / "rocket.toml")
args = parser.parse_args()

rocket = load_rocket(args.rocket)
flights = {name: simulate(rocket, make()) for name, make in CONTROLLERS.items()}

summaries = {name: summarize(rocket, f) for name, f in flights.items()}
print(f"\nRocket: {rocket.airframe.name}   Motor: {rocket.motor.name}")
print(f"{'':36s}" + "".join(f"{n:>14s}" for n in flights))
for key in next(iter(summaries.values())):
    print(f"{key:36s}" + "".join(f"{s[key]:14.2f}" for s in summaries.values()))

fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for i, (name, f) in enumerate(flights.items()):
    c = f"C{i}"
    ax[0, 0].plot(f.t, f.altitude, c, label=name)
    ax[0, 1].plot(f.t, f.mach, c, label=name)
    ax[0, 2].plot(f.t, np.degrees(f.roll_angle), c, label=name)
    ax[1, 0].plot(f.t, np.degrees(f.omega[:, 0]), c, label=name)
    ax[1, 1].plot(f.t, f.command, c, ls="--", alpha=0.5, label=f"{name}: command")
    ax[1, 1].plot(f.t, f.deflection, c, label=f"{name}: actual")
    ax[1, 2].plot(f.t, np.degrees(f.alpha), c, label=name)
ax[0, 0].set(title="Altitude", ylabel="m")
ax[0, 1].set(title="Mach number")
ax[0, 2].set(title="Roll angle", ylabel="deg")
ax[1, 0].set(title="Roll rate", ylabel="deg/s")
ax[1, 1].set(title="Fin deflection", ylabel="deg")
ax[1, 2].set(title="Angle of attack", ylabel="deg", ylim=(0, 20))
for a in ax.flat:
    a.set_xlabel("time [s]")
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
fig.tight_layout()
out = HERE / "output"
out.mkdir(exist_ok=True)
fig.savefig(out / "flight.png", dpi=110)
plt.show()
