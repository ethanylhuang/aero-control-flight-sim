"""Post-flight numbers worth looking at (OpenRocket-style summary)."""
import numpy as np

from .mass_properties import mass_properties


def static_margin_calibers(rocket, t: float, mach: float = 0.1) -> float:
    """(CP - CG) / diameter. Above ~1-2 is stable; negative means it will tumble."""
    mp = mass_properties(rocket.airframe, rocket.motor, rocket.motor_aft_position_m, t)
    return (rocket.aero.table("cp_m", mach) - mp.cg) / rocket.airframe.diameter_m


def summarize(rocket, flight) -> dict:
    i_apogee = int(np.argmax(flight.altitude))
    g = 9.80665
    return {
        "Static margin, liftoff [cal]": static_margin_calibers(rocket, 0.0),
        "Static margin, burnout [cal]": static_margin_calibers(rocket, rocket.motor.burn_time),
        "Rail exit speed [m/s]": flight.rail_exit_speed,
        "Max speed [m/s]": float(flight.airspeed.max()),
        "Max Mach": float(flight.mach.max()),
        "Max dynamic pressure [kPa]": float(flight.q.max()) / 1000,
        "Max acceleration [g]": float(np.linalg.norm(flight.accel, axis=1).max()) / g,
        "Apogee [m]": float(flight.altitude[i_apogee]),
        "Time to apogee [s]": float(flight.t[i_apogee]),
        "Max angle of attack, >20 m/s [deg]": float(np.degrees(flight.alpha[flight.airspeed > 20].max())),
        "Max |roll rate| [deg/s]": float(np.degrees(np.abs(flight.omega[:, 0]).max())),
        "Final roll angle [deg]": float(np.degrees(flight.roll_angle[-1])),
        "Max fin deflection used [deg]": float(np.abs(flight.deflection).max()),
    }
