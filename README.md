# readinsp

`readinsp` reads Insta360 `.insp` still-image files into NumPy arrays.

## Usage

```python
from readinsp import read_insp, read_metadata, read_preview

image = read_insp("example.insp")
print(image.shape, image.dtype)  # (3040, 6080, 3) uint8

preview = read_preview("example.insp")
print(preview.shape)  # (960, 1920, 3)

metadata = read_metadata("example.insp")
print(metadata.width, metadata.height, metadata.preview_size)
print(metadata.exif.get("Model"))
print(metadata.trailer.fields.get("camera_model"))
```

## Rectilinear views

`read_rectilinear` undistorts the two circular fisheye images, combines the two lenses, and renders a regular perspective view:

```python
from readinsp import read_rectilinear

view = read_rectilinear(
    "example.insp",
    spherical_angle=(0, 0),  # yaw, pitch in degrees
    hfov=90,
    vfov=60,
    width=1200,
)
print(view.shape)  # (693, 1200, 3)
```

Angle convention: yaw `0` points at the left fisheye lens, yaw `180` points at the right fisheye lens, and positive pitch looks up. `spherical_angle` may also include a third roll value: `(yaw, pitch, roll)`. When `height` is omitted, it is computed from the FOVs so the output has approximately square angular sampling near the view center. The default dual-fisheye model assumes a `190` degree fisheye per lens with an `8` degree blend near the stitch boundary. Both values can be overridden.

Gravity leveling is available for files with a valid EXIF MakerNote gravity vector:

```python
from readinsp import SphericalAngle, read_rectilinear

floor = read_rectilinear(
    "example.insp",
    spherical_angle=SphericalAngle(yaw=0, pitch=-90, gravity_level=True),
    hfov=90,
    vfov=70,
    width=1000,
)
```

With `gravity_level=True`, positive pitch looks opposite gravity and negative pitch looks along gravity. Yaw is still camera-relative because the inspected files do not include a usable magnetic-north heading.

Install for local development from the repository root with:

```bash
python -m pip install -e readinsp
```

## Limitations

This is a reverse-engineered parser based on samples from the OneR and OneRS cameras and might not work for other camera models. Rectilinear rendering is an approximate open implementation and will not match Insta360 Studio's proprietary stitching exactly, especially near the lens seam.