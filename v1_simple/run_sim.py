"""Run a whole flight instantly and plot the results.  Usage:  .venv/bin/python run_sim.py
Settings live in config.py.
"""
import numpy as np
import matplotlib.pyplot as plt

import config
from simple_sim.simulate import simulate

rocket = config.make_rocket()
motor = rocket.motor
flight = simulate(rocket, config.CONTROLLER, dt=config.TIME_STEP)

i_apogee = np.argmax(flight.altitude)
print(f"Motor:          {motor.name}  (impulse {motor.total_impulse:.0f} Ns, burn {motor.burn_time:.2f} s)")
print(f"Liftoff mass:   {flight.mass[0]:.3f} kg")
print(f"Max speed:      {flight.speed.max():.1f} m/s")
print(f"Max accel:      {np.linalg.norm(flight.acceleration, axis=1).max() / 9.81:.1f} g")
print(f"Apogee:         {flight.altitude[i_apogee]:.0f} m at t = {flight.t[i_apogee]:.1f} s")
print(f"Downrange:      {np.hypot(*flight.position[-1, :2]):.0f} m")
print(f"Flight time:    {flight.t[-1]:.1f} s")

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax[0, 0].plot(flight.t, flight.altitude)
ax[0, 0].set(title="Altitude", xlabel="time [s]", ylabel="m")
ax[0, 1].plot(flight.t, flight.speed)
ax[0, 1].set(title="Speed", xlabel="time [s]", ylabel="m/s")
ax[1, 0].plot(flight.t, flight.thrust)
ax[1, 0].set(title="Thrust", xlabel="time [s]", ylabel="N")
downrange = np.hypot(flight.position[:, 0], flight.position[:, 1])
ax[1, 1].plot(downrange, flight.altitude)
ax[1, 1].set(title="Trajectory", xlabel="downrange [m]", ylabel="altitude [m]")
for a in ax.flat:
    a.grid(alpha=0.3)
fig.tight_layout()
out = config.HERE / "output"
out.mkdir(exist_ok=True)
fig.savefig(out / "flight.png", dpi=120)
plt.show()
