from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from PIL import ExifTags, Image

from .core import read_insp, read_metadata


@dataclass(frozen=True)
class SphericalAngle:
    """Center of a rectilinear view on the stitched sphere, in degrees."""

    yaw: float
    pitch: float = 0.0
    roll: float = 0.0
    gravity_level: bool = False


@dataclass(frozen=True)
class LensCalibration:
    center_x: float
    center_y: float
    radius: float
    rotation_degrees: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class InspCalibration:
    lenses: Tuple[LensCalibration, LensCalibration]
    image_size: Tuple[int, int]
    raw: Optional[str] = None

    @classmethod
    def from_string(cls, calibration: str) -> "InspCalibration":
        values = [float(part) for part in calibration.split("_")]
        if len(values) < 16:
            raise ValueError("expected at least 16 values in the calibration string")

        # Observed format:
        # mode, lens0_y, lens0_x, lens0_r, lens0_rx, lens0_ry, lens0_roll,
        #       lens1_y, lens1_x, lens1_r, lens1_rx, lens1_ry, lens1_roll,
        #       image_width, image_height, ...
        lens0 = LensCalibration(
            center_x=values[2],
            center_y=values[1],
            radius=values[3],
            rotation_degrees=(values[4], values[5], values[6]),
        )
        lens1 = LensCalibration(
            center_x=values[8],
            center_y=values[7],
            radius=values[9],
            rotation_degrees=(values[10], values[11], values[12]),
        )
        return cls(
            lenses=(lens0, lens1),
            image_size=(int(round(values[13])), int(round(values[14]))),
            raw=calibration,
        )

    @classmethod
    def from_image_shape(cls, image_shape: Sequence[int]) -> "InspCalibration":
        height = int(image_shape[0])
        width = int(image_shape[1])
        radius = min(width / 4.0, height / 2.0)
        return cls(
            lenses=(
                LensCalibration(width / 4.0, height / 2.0, radius),
                LensCalibration(3.0 * width / 4.0, height / 2.0, radius),
            ),
            image_size=(width, height),
        )


def read_rectilinear(
    path: Union[str, Path],
    spherical_angle: Union[SphericalAngle, Sequence[float]],
    hfov: float,
    vfov: float,
    *,
    width: int = 1024,
    height: Optional[int] = None,
    fisheye_fov: float = 190.0,
    blend_width: float = 8.0,
    use_lens_roll: bool = False,
    gravity_vector: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Read an .insp file and render a regular rectilinear view.

    `spherical_angle` is `(yaw, pitch)` or `(yaw, pitch, roll)` in degrees.
    Yaw 0 points at the left lens optical axis, yaw 180 points at the right
    lens optical axis, and positive pitch looks upward.
    """

    path = Path(path)
    angle, gravity_level = _angle_spec(spherical_angle)
    if gravity_level and gravity_vector is None:
        gravity_vector = read_gravity_vector(path)
    if gravity_level and gravity_vector is None:
        raise ValueError("gravity-leveling requested, but no gravity vector was found")

    image = read_insp(path)
    metadata = read_metadata(path)
    calibration_text = None
    if metadata.trailer is not None:
        calibration_text = metadata.trailer.fields.get("calibration")

    calibration = (
        InspCalibration.from_string(calibration_text)
        if isinstance(calibration_text, str)
        else InspCalibration.from_image_shape(image.shape)
    )
    return render_rectilinear(
        image,
        calibration,
        SphericalAngle(*angle, gravity_level=gravity_level),
        hfov,
        vfov,
        width=width,
        height=height,
        fisheye_fov=fisheye_fov,
        blend_width=blend_width,
        use_lens_roll=use_lens_roll,
        gravity_vector=gravity_vector,
    )


def render_rectilinear(
    image: np.ndarray,
    calibration: Optional[InspCalibration],
    spherical_angle: Union[SphericalAngle, Sequence[float]],
    hfov: float,
    vfov: float,
    *,
    width: int = 1024,
    height: Optional[int] = None,
    fisheye_fov: float = 190.0,
    blend_width: float = 8.0,
    use_lens_roll: bool = False,
    gravity_vector: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Render a rectilinear view from a dual-fisheye image array."""

    if calibration is None:
        calibration = InspCalibration.from_image_shape(image.shape)

    angle, gravity_level = _angle_spec(spherical_angle)
    basis = _raw_basis()
    if gravity_level:
        if gravity_vector is None:
            raise ValueError("gravity_vector is required for gravity-leveled rendering")
        basis = _gravity_leveled_basis(gravity_vector)

    out_width, out_height = _output_size(width, height, hfov, vfov)
    directions = _view_directions(
        angle,
        hfov,
        vfov,
        out_width,
        out_height,
        basis,
    )
    sampled = _sample_dual_fisheye(
        image,
        calibration,
        directions,
        fisheye_fov=fisheye_fov,
        blend_width=blend_width,
        use_lens_roll=use_lens_roll,
    )
    return _restore_dtype(sampled, image.dtype)


def read_gravity_vector(path: Union[str, Path]) -> Optional[Tuple[float, float, float]]:
    """Return the normalized camera-space gravity/down vector, if present.

    The inspected .insp files store this in the first three underscore-separated
    numbers of the EXIF MakerNote. The vector is returned in the same raw camera
    coordinate system used by `render_rectilinear`.
    """

    with Image.open(path) as image:
        maker_note = image.getexif().get_ifd(ExifTags.IFD.Exif).get(0x927C)
    return _gravity_from_maker_note(maker_note)


def _output_size(
    width: int,
    height: Optional[int],
    hfov: float,
    vfov: float,
) -> Tuple[int, int]:
    if width <= 0:
        raise ValueError("width must be positive")
    _validate_fov(hfov, "hfov")
    _validate_fov(vfov, "vfov")

    if height is None:
        ratio = np.tan(np.deg2rad(vfov) / 2.0) / np.tan(np.deg2rad(hfov) / 2.0)
        height = max(1, int(round(width * ratio)))
    if height <= 0:
        raise ValueError("height must be positive")
    return int(width), int(height)


def _validate_fov(value: float, name: str) -> None:
    if not 0.0 < value < 180.0:
        raise ValueError(f"{name} must be between 0 and 180 degrees")


def _angle_spec(
    angle: Union[SphericalAngle, Sequence[float]],
) -> Tuple[Tuple[float, float, float], bool]:
    if isinstance(angle, SphericalAngle):
        return (angle.yaw, angle.pitch, angle.roll), angle.gravity_level
    if len(angle) == 2:
        return (float(angle[0]), float(angle[1]), 0.0), False
    if len(angle) == 3:
        return (float(angle[0]), float(angle[1]), float(angle[2])), False
    raise ValueError("spherical_angle must contain yaw/pitch or yaw/pitch/roll")


def _view_directions(
    angle: Tuple[float, float, float],
    hfov: float,
    vfov: float,
    width: int,
    height: int,
    basis: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    yaw, pitch, roll = np.deg2rad(angle)
    basis_right, basis_up, basis_forward = basis
    center = (
        np.sin(yaw) * np.cos(pitch) * basis_right
        + np.sin(pitch) * basis_up
        + np.cos(yaw) * np.cos(pitch) * basis_forward
    )
    center /= np.linalg.norm(center)

    if abs(float(np.dot(center, basis_up))) > 0.999:
        right = basis_right.copy()
    else:
        right = np.cross(basis_up, center)
        right /= np.linalg.norm(right)
    up = np.cross(center, right)
    up /= np.linalg.norm(up)

    if roll:
        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        rolled_right = cos_roll * right + sin_roll * up
        rolled_up = -sin_roll * right + cos_roll * up
        right, up = rolled_right, rolled_up

    xs = (
        (np.arange(width, dtype=np.float32) + 0.5) / float(width) * 2.0 - 1.0
    ) * np.tan(np.deg2rad(hfov) / 2.0)
    ys = (
        1.0 - (np.arange(height, dtype=np.float32) + 0.5) / float(height) * 2.0
    ) * np.tan(np.deg2rad(vfov) / 2.0)
    xx, yy = np.meshgrid(xs, ys)
    directions = (
        center[None, None, :]
        + xx[:, :, None] * right[None, None, :]
        + yy[:, :, None] * up[None, None, :]
    )
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    return directions.astype(np.float32, copy=False)


def _raw_basis() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )


def _gravity_leveled_basis(
    gravity_vector: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gravity = _normalize_vector(np.asarray(gravity_vector, dtype=np.float32))
    up = -gravity

    for anchor in _raw_basis()[::-1]:
        forward = anchor - float(np.dot(anchor, up)) * up
        norm = float(np.linalg.norm(forward))
        if norm > 1e-6:
            forward = forward / norm
            right = np.cross(up, forward)
            right /= np.linalg.norm(right)
            forward = np.cross(right, up)
            forward /= np.linalg.norm(forward)
            return (
                right.astype(np.float32),
                up.astype(np.float32),
                forward.astype(np.float32),
            )

    raise ValueError("could not construct a gravity-leveled basis")


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("gravity vector must contain three finite numbers")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        raise ValueError("gravity vector is too small")
    return vector / norm


def _gravity_from_maker_note(
    maker_note: object,
) -> Optional[Tuple[float, float, float]]:
    if not isinstance(maker_note, bytes):
        return None

    text = maker_note.split(b"\0", 1)[0].decode("ascii", "ignore")
    try:
        values = [float(part) for part in text.split("_")]
    except ValueError:
        return None
    if len(values) < 3:
        return None

    vector = np.asarray(values[:3], dtype=np.float32)
    if not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    if norm < 0.25:
        return None
    vector = vector / norm
    return tuple(float(value) for value in vector)


def _sample_dual_fisheye(
    image: np.ndarray,
    calibration: InspCalibration,
    directions: np.ndarray,
    *,
    fisheye_fov: float,
    blend_width: float,
    use_lens_roll: bool,
) -> np.ndarray:
    if image.ndim not in {2, 3}:
        raise ValueError("image must be a 2D or 3D array")
    if len(calibration.lenses) != 2:
        raise ValueError("expected calibration for exactly two lenses")
    if not 0.0 < fisheye_fov <= 360.0:
        raise ValueError("fisheye_fov must be between 0 and 360 degrees")
    if blend_width < 0.0:
        raise ValueError("blend_width must be non-negative")

    source = image[:, :, None] if image.ndim == 2 else image
    coordinates = [
        _fisheye_coordinates(
            directions,
            calibration.lenses[0],
            lens_index=0,
            fisheye_fov=fisheye_fov,
            use_lens_roll=use_lens_roll,
        ),
        _fisheye_coordinates(
            directions,
            calibration.lenses[1],
            lens_index=1,
            fisheye_fov=fisheye_fov,
            use_lens_roll=use_lens_roll,
        ),
    ]
    samples = [_bilinear_sample(source, x, y, valid) for x, y, valid, _ in coordinates]

    if blend_width > 0.0:
        weights = []
        blend = np.deg2rad(blend_width)
        for _, _, valid, theta in coordinates:
            half_fov = np.deg2rad(fisheye_fov) / 2.0
            weight = np.clip((half_fov - theta) / blend, 0.0, 1.0)
            weights.append(np.where(valid, np.maximum(weight, 1e-6), 0.0))
        total = weights[0] + weights[1]
        out = np.zeros_like(samples[0], dtype=np.float32)
        has_weight = total > 0.0
        out[has_weight] = (
            samples[0][has_weight] * weights[0][has_weight, None]
            + samples[1][has_weight] * weights[1][has_weight, None]
        ) / total[has_weight, None]
    else:
        valid0 = coordinates[0][2]
        valid1 = coordinates[1][2]
        score0 = np.where(valid0, np.cos(coordinates[0][3]), -np.inf)
        score1 = np.where(valid1, np.cos(coordinates[1][3]), -np.inf)
        use1 = score1 > score0
        out = np.where(use1[:, :, None], samples[1], samples[0])
        out[~(valid0 | valid1)] = 0.0

    return out[:, :, 0] if image.ndim == 2 else out


def _fisheye_coordinates(
    directions: np.ndarray,
    lens: LensCalibration,
    *,
    lens_index: int,
    fisheye_fov: float,
    use_lens_roll: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if lens_index == 0:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        axis = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        x_axis = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    y_axis = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    local_x = np.tensordot(directions, x_axis, axes=([2], [0]))
    local_y = np.tensordot(directions, y_axis, axes=([2], [0]))
    local_z = np.tensordot(directions, axis, axes=([2], [0]))
    local_z = np.clip(local_z, -1.0, 1.0)

    if use_lens_roll:
        roll = np.deg2rad(lens.rotation_degrees[2])
        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        rolled_x = cos_roll * local_x - sin_roll * local_y
        rolled_y = sin_roll * local_x + cos_roll * local_y
        local_x, local_y = rolled_x, rolled_y

    theta = np.arccos(local_z)
    half_fov = np.deg2rad(fisheye_fov) / 2.0
    sin_theta = np.sqrt(np.maximum(1.0 - local_z * local_z, 0.0))
    radius = lens.radius * theta / half_fov
    scale = np.divide(
        radius,
        sin_theta,
        out=np.zeros_like(radius, dtype=np.float32),
        where=sin_theta > 1e-7,
    )

    x = lens.center_x + local_x * scale
    y = lens.center_y + local_y * scale
    valid = theta <= half_fov + 1e-6
    return x, y, valid, theta


def _bilinear_sample(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    height, width = image.shape[:2]
    in_bounds = (x >= 0.0) & (x <= width - 1) & (y >= 0.0) & (y <= height - 1)
    valid = valid & in_bounds

    x0 = np.floor(np.clip(x, 0, width - 1)).astype(np.int64)
    y0 = np.floor(np.clip(y, 0, height - 1)).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)

    wx = (x - x0)[:, :, None]
    wy = (y - y0)[:, :, None]

    top = image[y0, x0].astype(np.float32) * (1.0 - wx) + image[y0, x1].astype(
        np.float32
    ) * wx
    bottom = image[y1, x0].astype(np.float32) * (1.0 - wx) + image[y1, x1].astype(
        np.float32
    ) * wx
    out = top * (1.0 - wy) + bottom * wy
    out[~valid] = 0.0
    return out


def _restore_dtype(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(image), info.min, info.max).astype(dtype)
    return image.astype(dtype, copy=False)
