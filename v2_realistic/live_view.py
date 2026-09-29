"""Watch the flight play out live, with the controller running inside the loop.

    .venv/bin/python v2_realistic/live_view.py               # example PID
    .venv/bin/python v2_realistic/live_view.py --controller none --speed 4
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from realistic_sim.config import load_rocket
from realistic_sim.controller import NoController, RollPIDController
from realistic_sim.simulate import Simulation

HERE = Path(__file__).parent
FPS = 30


def make_live_view(sim: Simulation, speed: float = 1.0, fps: int = FPS):
    """Returns (fig, update, anim). Keep a reference to anim or it gets garbage collected."""
    steps_per_frame = max(1, round(speed / fps / sim.dt))
    hist = {k: [] for k in ("t", "alt", "mach", "roll", "rate", "cmd", "defl", "aoa")}

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    lines = {
        "alt": ax[0, 0].plot([], [])[0],
        "mach": ax[0, 1].plot([], [])[0],
        "roll": ax[0, 2].plot([], [])[0],
        "rate": ax[1, 0].plot([], [])[0],
        "cmd": ax[1, 1].plot([], [], "--", alpha=0.5, label="command")[0],
        "defl": ax[1, 1].plot([], [], label="actual")[0],
        "aoa": ax[1, 2].plot([], [])[0],
    }
    ax[0, 0].set(title="Altitude", ylabel="m")
    ax[0, 1].set(title="Mach number")
    ax[0, 2].set(title="Roll angle", ylabel="deg")
    ax[1, 0].set(title="Roll rate", ylabel="deg/s")
    ax[1, 1].set(title="Fin deflection", ylabel="deg")
    ax[1, 1].legend(fontsize=8)
    ax[1, 2].set(title="Angle of attack", ylabel="deg")
    for a in ax.flat:
        a.set_xlabel("time [s]")
        a.grid(alpha=0.3)
    status = fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def update(_frame):
        for _ in range(steps_per_frame):
            sim.step()
        info = sim._info["aero"]
        hist["t"].append(sim.t)
        hist["alt"].append(sim.position[2])
        hist["mach"].append(info.mach)
        hist["roll"].append(np.degrees(sim.roll_angle))
        hist["rate"].append(np.degrees(sim.omega[0]))
        hist["cmd"].append(sim.command)
        hist["defl"].append(sim.deflection)
        hist["aoa"].append(np.degrees(info.alpha) if info.airspeed > 20 else 0.0)
        for key, line in lines.items():
            line.set_data(hist["t"], hist[key])
        for a in ax.flat:
            a.relim()
            a.autoscale_view()
        state = "APOGEE / LANDED" if sim.done else ("on rail" if sim.on_rail else "flying")
        status.set_text(f"t = {sim.t:5.1f} s   alt = {sim.position[2]:6.0f} m   Mach {info.mach:4.2f}   "
                        f"roll {hist['roll'][-1]:6.1f} deg   [{state}]")
        if sim.done:
            anim.event_source.stop()

    anim = FuncAnimation(fig, update, interval=1000 / fps, cache_frame_data=False)
    return fig, update, anim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rocket", default=HERE / "rockets" / "example" / "rocket.toml")
    parser.add_argument("--controller", choices=["pid", "none"], default="pid")
    parser.add_argument("--speed", type=float, default=1.0, help="1 = real time, 4 = 4x faster")
    args = parser.parse_args()

    controller = RollPIDController() if args.controller == "pid" else NoController()
    sim = Simulation(load_rocket(args.rocket), controller)
    fig, update, anim = make_live_view(sim, args.speed)
    plt.show()
