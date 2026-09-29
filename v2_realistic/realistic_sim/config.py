"""Loads a rocket description (a .toml file + .csv tables) into plain dataclasses.

The .toml is where the OpenRocket numbers go; see rockets/example/rocket.toml.
Relative paths inside the .toml are relative to the .toml file itself.
"""
from dataclasses import dataclass
from pathlib import Path
import tomllib
import numpy as np

from .motor import Motor, load_motor


@dataclass
class Table:
    """A lookup table keyed by Mach number (values are held constant outside the table's range).
    CSV format: '#' comment lines, a header row whose first column is "mach", then numeric rows."""
    mach: np.ndarray
    columns: dict

    @classmethod
    def load(cls, path):
        with open(path) as f:
            rows = [ln.strip().split(",") for ln in f if ln.strip() and not ln.startswith("#")]
        names = [n.strip() for n in rows[0]]
        data = np.array([[float(x) for x in r] for r in rows[1:]])
        return cls(data[:, 0], {n: data[:, i] for i, n in enumerate(names) if i > 0})

    def __call__(self, column: str, mach: float) -> float:
        return float(np.interp(mach, self.mach, self.columns[column]))


@dataclass
class Airframe:            # everything about the rocket EXCLUDING the motor
    name: str
    diameter_m: float
    length_m: float
    dry_mass_kg: float
    dry_cg_m: float                  # from nose tip
    dry_roll_inertia_kgm2: float     # about the long axis
    dry_pitch_inertia_kgm2: float    # about the CG, perpendicular to the long axis


@dataclass
class Aero:
    table: Table                     # columns: cd_coast, cd_boost, cn_alpha, cp_m
    roll_damping_coeff: float        # Cl_p, negative
    roll_disturbance_coeff: float    # roll moment coefficient with the fins centered


@dataclass
class ControlFins:
    table: Table                     # column: roll_coeff_per_deg
    max_deflection_deg: float
    max_rate_deg_s: float


@dataclass
class SensorNoise:
    gyro_noise_rad_s: float
    gyro_bias_rad_s: float
    accel_noise_ms2: float
    baro_noise_m: float


@dataclass
class Launch:
    site_elevation_m: float
    temperature_offset_k: float
    angle_deg: float
    heading_deg: float
    rail_length_m: float


@dataclass
class WindConfig:
    speed_ms: float
    from_direction_deg: float
    reference_height_m: float
    shear_exponent: float


@dataclass
class Settings:
    time_step_s: float
    control_rate_hz: float
    random_seed: int
    stop_at_apogee: bool
    max_time_s: float


@dataclass
class RocketConfig:
    airframe: Airframe
    motor: Motor
    motor_aft_position_m: float      # nose tip to the aft end of the motor
    aero: Aero
    control_fins: ControlFins
    sensors: SensorNoise
    launch: Launch
    wind: WindConfig
    settings: Settings


def load_rocket(toml_path) -> RocketConfig:
    toml_path = Path(toml_path)
    base = toml_path.parent
    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)

    aero = dict(raw["aero"])
    fins = dict(raw["control_fins"])
    return RocketConfig(
        airframe=Airframe(**raw["airframe"]),
        motor=load_motor(base / raw["motor"]["file"]),
        motor_aft_position_m=raw["motor"]["aft_position_m"],
        aero=Aero(Table.load(base / aero.pop("table")), **aero),
        control_fins=ControlFins(Table.load(base / fins.pop("table")), **fins),
        sensors=SensorNoise(**raw["sensors"]),
        launch=Launch(**raw["launch"]),
        wind=WindConfig(**raw["wind"]),
        settings=Settings(**raw["simulation"]),
    )
