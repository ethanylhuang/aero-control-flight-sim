"""PHYSICS: the 6-degree-of-freedom equations of motion (Newton + Euler).

State vector s (14 numbers):
    s[0:3]   position in world frame [m]      (x east, y north, z up, origin at the pad)
    s[3:6]   velocity in world frame [m/s]
    s[6:10]  attitude quaternion (w, x, y, z), rotates body -> world
    s[10:13] angular velocity in BODY frame [rad/s]   (roll rate p, then q, r)
    s[13]    roll angle [rad], the integral of the roll rate p

    Translation:  m dv/dt = R (thrust + aerodynamic force) + m g
    Rotation:     I dw/dt = M - w x (I w)             (Euler's equation, I = diag(Ixx, Iyy, Iyy))
"""
from dataclasses import dataclass
import numpy as np

from .aerodynamics import aero_loads
from .atmosphere import Atmosphere, Wind, gravity
from .mass_properties import mass_properties
from .integrator import quat_to_matrix, quat_multiply

POS, VEL, QUAT, OMEGA, ROLL = slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13), 13


@dataclass
class Environment:
    atmosphere: Atmosphere
    wind: Wind


class Dynamics:
    def __init__(self, rocket, env: Environment):
        self.rocket = rocket
        self.env = env

    def derivatives(self, t, s, deflection_deg, on_rail):
        """Returns (ds/dt, info). `info` holds extras for logging and sensors."""
        rk = self.rocket
        pos, vel, quat, omega = s[POS], s[VEL], s[QUAT], s[OMEGA]
        R = quat_to_matrix(quat)

        mp = mass_properties(rk.airframe, rk.motor, rk.motor_aft_position_m, t)
        thrust = rk.motor.thrust(t)
        rho, sound = self.env.atmosphere.properties(pos[2])
        wind = self.env.wind.vector(pos[2])

        v_body = R.T @ (vel - wind)
        f_aero, m_aero, aero = aero_loads(v_body, omega, mp.cg, rho, sound, thrust > 0.0,
                                          deflection_deg, rk.airframe, rk.aero, rk.control_fins)
        f_body = f_aero + np.array([thrust, 0.0, 0.0])

        g = gravity(pos[2] + self.env.atmosphere.ground_elevation_m)
        accel = R @ f_body / mp.mass + np.array([0.0, 0.0, -g])

        ds = np.zeros(14)
        ds[POS] = vel

        if on_rail:
            # Guided along the rail: only motion along the rail axis, no rotation.
            axis = R[:, 0]
            a_along = float(accel @ axis)
            if a_along < 0.0 and float(vel @ axis) <= 0.0:
                a_along = 0.0              # not enough thrust to lift off yet: held by the rail
            ds[VEL] = a_along * axis
        else:
            inertia = np.array([mp.ixx, mp.iyy, mp.iyy])
            ds[VEL] = accel
            ds[QUAT] = 0.5 * quat_multiply(quat, np.array([0.0, *omega]))
            ds[OMEGA] = (m_aero - np.cross(omega, inertia * omega)) / inertia
            ds[ROLL] = omega[0]

        info = {"aero": aero, "mass": mp.mass, "cg": mp.cg, "thrust": thrust,
                "specific_force": f_body / mp.mass}   # what an accelerometer would feel (body frame)
        return ds, info
