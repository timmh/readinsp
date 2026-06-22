from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from readinsp import (
    InspCalibration,
    InspFormatError,
    LensCalibration,
    SphericalAngle,
    extract_preview_jpeg,
    read_gravity_vector,
    read_insp,
    read_metadata,
    read_preview,
    render_rectilinear,
    trailer_bytes,
)


def _jpeg(width, height, color):
    image = Image.new("RGB", (width, height), color)
    out = BytesIO()
    image.save(out, format="JPEG")
    return out.getvalue()


def _insp_like_file(tmp_path):
    main = _jpeg(8, 4, (10, 20, 30))
    preview = _jpeg(4, 2, (40, 50, 60))
    app2 = b"\xff\xe2" + (len(preview) + 2).to_bytes(2, "big") + preview
    path = tmp_path / "sample.insp"
    path.write_bytes(main[:2] + app2 + main[2:] + b"trailer")
    return path, preview


def test_read_insp_reads_primary_jpeg(tmp_path):
    path, _ = _insp_like_file(tmp_path)

    arr = read_insp(path)

    assert arr.shape == (4, 8, 3)
    assert arr.dtype == np.uint8


def test_read_preview_reads_app2_jpeg(tmp_path):
    path, preview = _insp_like_file(tmp_path)

    assert extract_preview_jpeg(path) == preview
    arr = read_preview(path)

    assert arr.shape == (2, 4, 3)
    assert arr.dtype == np.uint8


def test_read_metadata_reports_structure(tmp_path):
    path, preview = _insp_like_file(tmp_path)

    metadata = read_metadata(path)

    assert metadata.width == 8
    assert metadata.height == 4
    assert metadata.preview_size == (4, 2)
    assert metadata.preview_bytes == len(preview)
    assert metadata.trailer_bytes == len(b"trailer")
    assert trailer_bytes(path) == b"trailer"


def test_missing_preview_raises(tmp_path):
    path = tmp_path / "no-preview.insp"
    path.write_bytes(_jpeg(2, 2, (0, 0, 0)))

    with pytest.raises(InspFormatError):
        extract_preview_jpeg(path)


def test_parse_insta360_calibration_string():
    calibration = InspCalibration.from_string(
        "2_1483.49_1520.8_1528.26_-0.387328_0.0854089_-179.779_"
        "1481.97_4560.31_1522.65_1.0036_0.113175_0.521564_6080_3040_3105"
    )

    assert calibration.image_size == (6080, 3040)
    assert calibration.lenses[0].center_x == pytest.approx(1520.8)
    assert calibration.lenses[0].center_y == pytest.approx(1483.49)
    assert calibration.lenses[1].center_x == pytest.approx(4560.31)
    assert calibration.lenses[1].center_y == pytest.approx(1481.97)


def test_render_rectilinear_combines_two_lenses():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :100] = (255, 0, 0)
    image[:, 100:] = (0, 0, 255)
    calibration = InspCalibration(
        lenses=(
            LensCalibration(center_x=50, center_y=50, radius=50),
            LensCalibration(center_x=150, center_y=50, radius=50),
        ),
        image_size=(200, 100),
    )

    front = render_rectilinear(
        image, calibration, (0, 0), hfov=30, vfov=30, width=20, height=10
    )
    back = render_rectilinear(
        image, calibration, (180, 0), hfov=30, vfov=30, width=20, height=10
    )

    assert front.shape == (10, 20, 3)
    assert back.shape == (10, 20, 3)
    assert front[..., 0].mean() > 240
    assert front[..., 2].mean() < 15
    assert back[..., 2].mean() > 240
    assert back[..., 0].mean() < 15


def test_gravity_leveled_rendering_uses_gravity_vector():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :100] = (255, 0, 0)
    image[:, 100:] = (0, 0, 255)
    calibration = InspCalibration(
        lenses=(
            LensCalibration(center_x=50, center_y=50, radius=50),
            LensCalibration(center_x=150, center_y=50, radius=50),
        ),
        image_size=(200, 100),
    )

    down = render_rectilinear(
        image,
        calibration,
        SphericalAngle(0, -90, gravity_level=True),
        hfov=30,
        vfov=30,
        width=20,
        height=10,
        gravity_vector=(0, 0, 1),
    )
    up = render_rectilinear(
        image,
        calibration,
        SphericalAngle(0, 90, gravity_level=True),
        hfov=30,
        vfov=30,
        width=20,
        height=10,
        gravity_vector=(0, 0, 1),
    )

    assert down[..., 0].mean() > 240
    assert down[..., 2].mean() < 15
    assert up[..., 2].mean() > 240
    assert up[..., 0].mean() < 15


def test_read_gravity_vector_returns_none_without_maker_note(tmp_path):
    path = tmp_path / "no-maker-note.jpg"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(path)

    assert read_gravity_vector(path) is None
