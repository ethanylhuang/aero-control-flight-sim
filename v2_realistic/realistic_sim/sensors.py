"""Sensor model: turns true simulation values into noisy readings for the controller."""
import numpy as np

from .controller import SensorReadings


class SensorSuite:
    def __init__(self, noise, seed: int):
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.gyro_bias = self.rng.normal(0.0, noise.gyro_bias_rad_s, 3)   # constant offset for the whole flight

    def read(self, t, omega, specific_force, altitude) -> SensorReadings:
        n, rng = self.noise, self.rng
        return SensorReadings(
            t=t,
            gyro=omega + self.gyro_bias + rng.normal(0.0, n.gyro_noise_rad_s, 3),
            accel=specific_force + rng.normal(0.0, n.accel_noise_ms2, 3),
            baro_altitude=altitude + rng.normal(0.0, n.baro_noise_m),
        )
