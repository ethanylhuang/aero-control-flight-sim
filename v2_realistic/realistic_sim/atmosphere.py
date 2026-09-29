"""PHYSICS: air (density, speed of sound), wind, gravity.

International Standard Atmosphere up to 20 km, plus an optional temperature offset
(e.g. a hot launch day) and a power-law wind profile.
"""
from dataclasses import dataclass
import numpy as np

R_AIR = 287.053     # J/(kg K)
GAMMA = 1.4
G0 = 9.80665        # m/s^2
R_EARTH = 6_371_000.0
T0, P0 = 288.15, 101_325.0
LAPSE = 0.0065      # K/m in the troposphere
H_TROPOPAUSE = 11_000.0


def gravity(altitude_asl: float) -> float:
    return G0 * (R_EARTH / (R_EARTH + altitude_asl)) ** 2


@dataclass
class Atmosphere:
    ground_elevation_m: float = 0.0
    temperature_offset_k: float = 0.0

    def properties(self, altitude_agl: float):
        """Returns (air density [kg/m^3], speed of sound [m/s]) at a height above the launch site."""
        h = float(np.clip(altitude_agl + self.ground_elevation_m, 0.0, 20_000.0))
        if h <= H_TROPOPAUSE:
            temp = T0 - LAPSE * h
            pressure = P0 * (temp / T0) ** (G0 / (R_AIR * LAPSE))
        else:
            temp = T0 - LAPSE * H_TROPOPAUSE
            p11 = P0 * (temp / T0) ** (G0 / (R_AIR * LAPSE))
            pressure = p11 * np.exp(-G0 * (h - H_TROPOPAUSE) / (R_AIR * temp))
        temp += self.temperature_offset_k
        density = pressure / (R_AIR * temp)
        return density, float(np.sqrt(GAMMA * R_AIR * temp))


@dataclass
class Wind:
    speed_ms: float = 0.0             # speed at reference height
    from_direction_deg: float = 0.0   # compass direction the wind blows FROM (0 = north, 90 = east)
    reference_height_m: float = 10.0
    shear_exponent: float = 0.14      # wind speeds up with height as (h/h_ref)^exponent

    def vector(self, altitude_agl: float) -> np.ndarray:
        """Wind velocity in world coordinates (x = east, y = north, z = up) [m/s]."""
        h = max(altitude_agl, 1.0)
        speed = self.speed_ms * (h / self.reference_height_m) ** self.shear_exponent
        blowing_toward = np.radians(self.from_direction_deg + 180.0)
        return speed * np.array([np.sin(blowing_toward), np.cos(blowing_toward), 0.0])
