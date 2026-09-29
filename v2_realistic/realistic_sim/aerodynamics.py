"""PHYSICS: aerodynamic forces and moments on the rocket.

Body frame: x along the rocket toward the nose, y and z sideways.
Positions along the rocket are measured from the nose tip, pointing aft.

Model (same idea as OpenRocket / Barrowman):
  * Axial drag       F = -q A Cd(Mach)                 (Cd differs while the motor burns)
  * Normal force     N =  q A CNalpha(Mach) sin(alpha) (acts at the center of pressure, CP)
        -> because CP is behind the CG this makes the rocket turn into the wind (weathercock).
        -> the flow angle is evaluated AT THE CP, including the rocket's own rotation,
           which gives pitch/yaw damping for free.
  * Roll moment      M_x = q A d [ Cl0  +  Clp * p d / (2V)  +  Cl_delta(Mach) * delta ]
                            disturbance   roll damping        control fins
Valid for small-to-moderate angle of attack (a stable rocket in flight).
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class AeroInfo:
    q: float = 0.0        # dynamic pressure [Pa]
    mach: float = 0.0
    alpha: float = 0.0    # total angle of attack [rad]
    airspeed: float = 0.0


def aero_loads(v_body, omega, cg, rho, speed_of_sound, powered, deflection_deg, airframe, aero, fins):
    """Force [N] and moment about the CG [N m], both in the body frame, plus a little diagnostic info."""
    V = float(np.linalg.norm(v_body))
    if V < 0.5:                      # essentially no airflow yet (sitting on the pad)
        return np.zeros(3), np.zeros(3), AeroInfo()

    d = airframe.diameter_m
    area = np.pi * d**2 / 4
    q = 0.5 * rho * V**2
    mach = V / speed_of_sound

    cd = aero.table("cd_boost" if powered else "cd_coast", mach)
    cn_alpha = aero.table("cn_alpha", mach)
    r_cp = np.array([cg - aero.table("cp_m", mach), 0.0, 0.0])   # CP position relative to CG (body x)

    # --- axial drag ---
    force = np.array([-np.sign(v_body[0]) * q * area * cd, 0.0, 0.0])

    # --- normal force at the CP ---
    v_cp = v_body + np.cross(omega, r_cp)          # air velocity seen at the CP, includes rotation
    v_side = np.array([0.0, v_cp[1], v_cp[2]])
    side = float(np.linalg.norm(v_side))
    alpha = float(np.arctan2(side, v_cp[0]))
    normal = np.zeros(3)
    if side > 1e-9:
        normal = -q * area * cn_alpha * np.sin(alpha) * v_side / side
    force += normal
    moment = np.cross(r_cp, normal)

    # --- roll ---
    cl = (aero.roll_disturbance_coeff
          + aero.roll_damping_coeff * omega[0] * d / (2 * V)
          + fins.table("roll_coeff_per_deg", mach) * deflection_deg)
    moment[0] += q * area * d * cl

    return force, moment, AeroInfo(q, mach, alpha, V)
