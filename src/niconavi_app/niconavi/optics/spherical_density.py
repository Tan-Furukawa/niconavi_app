from __future__ import annotations

import numpy as np


def make_spherical_kde_on_stereographic_grid(
    inclination_rad: np.ndarray,
    azimuth_rad: np.ndarray,
    *,
    n_grid: int,
    bandwidth_deg: float,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bandwidth_deg <= 0:
        raise ValueError("bandwidth_deg must be positive.")

    x_grid, y_grid = np.meshgrid(
        np.linspace(-1.0, 1.0, n_grid),
        np.linspace(-1.0, 1.0, n_grid),
    )
    valid_grid = x_grid**2 + y_grid**2 < 1.0
    grid_vectors = _stereographic_xy_to_vectors(x_grid[valid_grid], y_grid[valid_grid])
    sample_vectors = _angles_to_vectors(inclination_rad, azimuth_rad)

    bandwidth_rad = np.radians(bandwidth_deg)
    kappa = 1.0 / (bandwidth_rad * bandwidth_rad)

    density_values = np.zeros(len(grid_vectors), dtype=np.float64)
    for start in range(0, len(grid_vectors), chunk_size):
        stop = min(start + chunk_size, len(grid_vectors))
        dots = np.clip(grid_vectors[start:stop] @ sample_vectors.T, -1.0, 1.0)
        density_values[start:stop] = np.mean(np.exp(kappa * (dots - 1.0)), axis=1)

    density = np.full(x_grid.shape, np.nan, dtype=np.float64)
    density[valid_grid] = density_values
    density_sum = np.nansum(density)
    if density_sum > 0:
        density /= density_sum

    return x_grid, y_grid, density


def _angles_to_vectors(inclination_rad: np.ndarray, azimuth_rad: np.ndarray) -> np.ndarray:
    sin_theta = np.sin(inclination_rad)
    return np.column_stack(
        [
            sin_theta * np.cos(azimuth_rad),
            sin_theta * np.sin(azimuth_rad),
            np.cos(inclination_rad),
        ]
    )


def _stereographic_xy_to_vectors(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    radius = np.sqrt(x * x + y * y)
    theta = 2.0 * np.arctan(radius)
    azimuth = np.arctan2(y, x)
    return _angles_to_vectors(theta, azimuth)
