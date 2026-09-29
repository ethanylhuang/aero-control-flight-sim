# Rocket Aerodynamic Control Flight Simulator

Two completely separate simulators. They share only the thrust-curve data in `data/`.

| | **v1_simple** | **v2_realistic** |
|---|---|---|
| Purpose | What was asked for first: basic point-mass physics | Predict the real flight, including roll control |
| Model | Thrust + gravity, `F = ma`, integrate twice | Full 6-DOF: air, drag, stability, wind, rail, roll, sensors, servo |
| Inputs | Mass, launch angle, thrust curve | An OpenRocket-derived `rocket.toml` + CSV tables |
| Roll / controller | Controller hook exists, not used yet | Roll dynamics, CFD fin table, sensors, servo limits, example PID |

## Setup
```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## v1_simple
```
.venv/bin/python v1_simple/run_sim.py      # whole flight + plots
.venv/bin/python v1_simple/live_view.py    # live playback
```
Settings in `v1_simple/config.py`. The physics is one small file: `v1_simple/simple_sim/physics.py`.

## v2_realistic
```
.venv/bin/python v2_realistic/run_flight.py                    # with vs without controller, OpenRocket-style summary
.venv/bin/python v2_realistic/live_view.py --speed 4           # live playback (--controller none|pid)
.venv/bin/python v2_realistic/run_flight.py --rocket my_rocket/rocket.toml
```
All rocket data lives in `v2_realistic/rockets/example/` (copy the folder and edit it):
- `rocket.toml`: masses, inertia, CG, launch settings, wind, sensor noise, servo limits. Every value is commented with where to get it.
- `aero.csv`: Cd, CN-alpha, CP versus Mach (from OpenRocket).
- `control_fins.csv`: the CFD table, roll-moment coefficient per degree of deflection.

**All numbers in the example rocket are placeholders.**

| Layer | Files |
|---|---|
| **Physics** (read these to check the science) | `atmosphere.py`, `mass_properties.py`, `aerodynamics.py`, `dynamics.py` |
| Backend (no physics) | `integrator.py` (RK4, quaternions), `simulate.py` (loop), `config.py`, `motor.py`, `sensors.py`, `actuator.py`, `analysis.py` |
| Controller plug-in | `controller.py`: subclass `Controller`, return a fin deflection from noisy sensor readings |

What v2 models: ISA atmosphere and wind profile, mass/CG/inertia changing as the motor burns, drag (Mach-dependent, different while burning), normal force at the CP (weathercocking + pitch damping), launch rail, roll disturbance + roll damping + fin control moment, gyro/accelerometer/barometer noise, servo rate and travel limits, controller running at its own rate (default 100 Hz) inside a faster physics loop (1 kHz).

What v2 does NOT model yet: parachute/descent (stops at apogee), thrust misalignment, turbulence/gusts, jet damping, fin flutter or structural flex, Earth rotation, pitch/yaw control, and accuracy beyond the linear normal-force region (large angle of attack).

## Tests
```
.venv/bin/python -m pytest v1_simple v2_realistic
```
