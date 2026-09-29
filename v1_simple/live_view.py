"""Watch the flight play out live.  Usage:  .venv/bin/python live_view.py

The simulation is stepped a little each animation frame, so the controller runs
live inside the loop. Settings live in config.py; SPEED below is playback speed.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import config
from simple_sim.simulate import Simulation

SPEED = 1.0   # 1.0 = real time, 4.0 = 4x faster
FPS = 30


def make_live_view(sim: Simulation, speed: float = SPEED, fps: int = FPS):
    """Build the figure. Returns (fig, update, anim). Keep a reference to anim or it gets garbage collected."""
    # Downsampled history, appended once per frame (not per sim step) so plotting stays cheap.
    hist = {"t": [], "alt": [], "speed": [], "range": [], "fin": []}
    steps_per_frame = max(1, round(speed / fps / sim.dt))

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    (l_alt,) = ax[0, 0].plot([], [])
    ax[0, 0].set(title="Altitude", xlabel="time [s]", ylabel="m")
    (l_spd,) = ax[0, 1].plot([], [])
    ax[0, 1].set(title="Speed", xlabel="time [s]", ylabel="m/s")
    (l_fin,) = ax[1, 0].plot([], [])
    ax[1, 0].set(title="Controller output (fin deflection)", xlabel="time [s]", ylabel="deg")
    (l_traj,) = ax[1, 1].plot([], [])
    (dot,) = ax[1, 1].plot([], [], "ro")
    ax[1, 1].set(title="Trajectory", xlabel="downrange [m]", ylabel="altitude [m]")
    for a in ax.flat:
        a.grid(alpha=0.3)
    status = fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def update(_frame):
        for _ in range(steps_per_frame):
            sim.step()

        p = sim.position
        hist["t"].append(sim.t)
        hist["alt"].append(p[2])
        hist["speed"].append(np.linalg.norm(sim.velocity))
        hist["range"].append(np.hypot(p[0], p[1]))
        hist["fin"].append(sim.fin_deflection)

        l_alt.set_data(hist["t"], hist["alt"])
        l_spd.set_data(hist["t"], hist["speed"])
        l_fin.set_data(hist["t"], hist["fin"])
        l_traj.set_data(hist["range"], hist["alt"])
        dot.set_data([hist["range"][-1]], [hist["alt"][-1]])
        for a in ax.flat:
            a.relim()
            a.autoscale_view()

        state = "LANDED" if sim.done else "flying"
        status.set_text(f"t = {sim.t:5.1f} s   alt = {p[2]:6.0f} m   speed = {hist['speed'][-1]:5.0f} m/s   [{state}]")
        if sim.done:
            anim.event_source.stop()

    anim = FuncAnimation(fig, update, interval=1000 / fps, cache_frame_data=False)
    return fig, update, anim


if __name__ == "__main__":
    sim = Simulation(config.make_rocket(), config.CONTROLLER, dt=config.TIME_STEP)
    fig, update, anim = make_live_view(sim)
    plt.show()
