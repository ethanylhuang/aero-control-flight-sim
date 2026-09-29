"""PHYSICS: mass, center of gravity and moments of inertia as propellant burns.

Rocket-without-motor numbers come from OpenRocket. The motor is modelled as a
uniform cylinder that loses mass as it burns; its contribution is added with
the parallel-axis theorem. Positions are measured from the nose tip, pointing aft.
"""
from dataclasses import dataclass


@dataclass
class MassProperties:
    mass: float   # kg
    cg: float     # m from nose tip
    ixx: float    # roll moment of inertia [kg m^2]
    iyy: float    # pitch/yaw moment of inertia about the CG [kg m^2]


def mass_properties(airframe, motor, motor_aft_position_m: float, t: float) -> MassProperties:
    m_d, cg_d = airframe.dry_mass_kg, airframe.dry_cg_m

    m_m = motor.mass(t)
    length, radius = motor.length_m, motor.diameter_m / 2
    cg_m = motor_aft_position_m - length / 2

    mass = m_d + m_m
    cg = (m_d * cg_d + m_m * cg_m) / mass

    # Solid cylinder: roll = 1/2 m r^2, transverse = m (3 r^2 + L^2) / 12
    ixx = airframe.dry_roll_inertia_kgm2 + 0.5 * m_m * radius**2
    iyy = (airframe.dry_pitch_inertia_kgm2 + m_d * (cg_d - cg) ** 2
           + m_m * (3 * radius**2 + length**2) / 12 + m_m * (cg_m - cg) ** 2)
    return MassProperties(mass, cg, ixx, iyy)
