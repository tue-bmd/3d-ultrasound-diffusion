"""Loading of 3D ultrasound volumes from zea-format hdf5 files.

Every file holds a 3D image sequence in ``data/image``:

- ``values``: ``(n_frames, depth, azimuth, elevation)``, uint8 in ``[0, 255]`` or float32 in dB.
- ``coordinates``: the Cartesian ``(x, y, z)`` position in metres of every sample.

For polar (frustum) volumes, i.e. sampled on a ``(rho, theta, phi)`` grid, the frustum is
recovered from the coordinates and used for scan conversion when plotting. Volumes on a
Cartesian grid (already scan-converted) are used as is.

Paths can be local files / folders or ``hf://<repo_id>/<path>`` on the Hugging Face Hub.

Volumes are returned as ``(frames, azimuth, depth, elevation)``: every ``[:, a]`` slice is one
B-plane (depth x elevation), which is what the 2D diffusion prior models.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import zea
from zea.display import frustum_convert_xyz2rtp

VALUES = "data/image/values"
COORDINATES = "data/image/coordinates"

PathLike = Union[str, Path]


@dataclass
class Volume:
    """A sequence of 3D volumes, ``data`` has shape ``(frames, azimuth, depth, elevation)``."""

    data: np.ndarray
    image_range: Tuple[float, float]
    grid: Optional[dict] = None
    """Frustum ``{rho [mm], theta, phi}_range`` for scan conversion, None if not polar."""
    path: Optional[str] = None


def frustum_ranges(coordinates, atol=1e-4) -> Optional[dict]:
    """Frustum ranges if ``coordinates`` (depth, azimuth, elevation, 3) lie on a polar grid.

    rho must only vary along depth, theta along azimuth and phi along elevation.
    """
    x, y, z = np.moveaxis(np.asarray(coordinates, dtype=np.float64), -1, 0)
    if np.mean(z <= 0) > 0.5:
        return None
    limits = (-np.pi / 2, np.pi / 2)
    rho, theta, phi = (np.asarray(v) for v in frustum_convert_xyz2rtp(x, y, z, limits, limits))
    apex = z <= 0  # angles are undefined at (or behind) the apex
    theta, phi = np.where(apex, np.nan, theta), np.where(apex, np.nan, phi)
    for values, axis in ((rho, 0), (theta, 1), (phi, 2)):
        other = tuple(a for a in range(3) if a != axis)
        spread = np.nanmax(values, axis=other) - np.nanmin(values, axis=other)
        if np.nanmax(spread) > atol:
            return None
    return {
        "rho_range": (float(rho.min() * 1e3), float(rho.max() * 1e3)),
        "theta_range": (float(np.nanmin(theta)), float(np.nanmax(theta))),
        "phi_range": (float(np.nanmin(phi)), float(np.nanmax(phi))),
    }


def value_range(dtype, db_range: Tuple[float, float] = (-60.0, 0.0)) -> Tuple[float, float]:
    """Value range of stored images: uint8 in [0, 255], float32 in dB (clipped to ``db_range``)."""
    return (0.0, 255.0) if np.dtype(dtype) == np.uint8 else tuple(db_range)


def load_volume(
    path: PathLike,
    n_frames: Optional[int] = None,
    start: int = 0,
    db_range: Tuple[float, float] = (-60.0, 0.0),
    with_grid: bool = True,
) -> Volume:
    """Load ``n_frames`` frames (all if None) of a volume sequence, starting at frame ``start``.

    Args:
        path: local path or ``hf://`` path to a zea hdf5 file.
        n_frames: number of frames to load (all if None).
        start: first frame to load.
        db_range: dynamic range for images stored in dB (float32).
        with_grid: also read the coordinates to recover the frustum (for scan conversion).
    """
    with zea.File(str(path), "r") as f:
        dataset = f.dataset(VALUES)
        if n_frames is not None and dataset.shape[0] - start < n_frames:
            raise ValueError(f"{path} has {dataset.shape[0]} frames, {n_frames} requested.")
        stop = None if n_frames is None else start + n_frames
        values = np.asarray(dataset[start:stop])
        coordinates = f.dataset(COORDINATES)[()] if with_grid else None
    if values.ndim != 4:
        raise ValueError(f"Expected {VALUES} of shape (frames, z, x, y), got {values.shape}")

    image_range = value_range(values.dtype, db_range)
    values = np.clip(values, *image_range)
    grid = None
    if coordinates is not None:
        grid = frustum_ranges(coordinates[0] if coordinates.ndim == 5 else coordinates)
    # (frames, depth, azimuth, elevation) -> (frames, azimuth, depth, elevation)
    data = np.transpose(values.astype("float32"), (0, 2, 1, 3))
    return Volume(data, image_range, grid, str(path))


def list_files(folder: PathLike) -> List[str]:
    """Sorted hdf5 files in a local folder / file, or ``hf://<repo_id>[/<subdir>]``."""
    return sorted(str(p) for p in zea.Dataset(str(folder), validate=False).file_paths)


def resolve_file(data: PathLike, file: Optional[str] = None, seed: Optional[int] = None) -> str:
    """Pick ``file`` (relative to ``data``), ``data`` itself if it is a file, or a random file."""
    if file is not None:
        if file.startswith("hf://") or Path(file).is_absolute():
            return file
        return f"{str(data).rstrip('/')}/{file}"
    files = list_files(data)
    if not files:
        raise FileNotFoundError(f"No hdf5 files found in {data}")
    return random.Random(seed).choice(files)
