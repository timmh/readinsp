from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Optional, Tuple, Union

import numpy as np
from PIL import ExifTags, Image, ImageOps

JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"

APP1 = 0xE1
APP2 = 0xE2
SOS = 0xDA

SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}

TRAILER_FIELD_NAMES = {
    1: "serial",
    2: "camera_model",
    3: "firmware",
    5: "calibration",
    9: "jpeg_end_offset_or_size",
    10: "unknown_10",
    19: "image_size_message",
    24: "unknown_24",
    25: "unknown_25",
    26: "source_path_message",
    27: "dimension_message",
    31: "orientation_or_pose",
    65: "thumbnail_size_message",
}


class InspFormatError(ValueError):
    """Raised when a file does not look like the observed .insp format."""


@dataclass(frozen=True)
class JPEGSegment:
    marker: int
    offset: int
    payload: bytes


@dataclass(frozen=True)
class InspTrailer:
    raw: bytes
    prefix: bytes
    fields: dict[str, Any]
    decoded_until: int


@dataclass(frozen=True)
class InspMetadata:
    path: Path
    width: int
    height: int
    precision: int
    components: int
    preview_size: Optional[Tuple[int, int]]
    preview_bytes: int
    trailer_bytes: int
    exif: dict[str, Any]
    trailer: Optional[InspTrailer]


def read_insp(
    path: Union[str, Path, BinaryIO],
    *,
    mode: Optional[str] = "RGB",
    apply_exif_orientation: bool = False,
) -> np.ndarray:
    """Read the primary .insp image as a NumPy array.

    Parameters
    ----------
    path:
        File path or binary file object.
    mode:
        Optional Pillow mode conversion. The default returns RGB arrays.
        Pass `None` to keep Pillow's decoded mode.
    apply_exif_orientation:
        Apply Pillow's EXIF orientation transform before conversion.
    """

    return _decode_image(path, mode=mode, apply_exif_orientation=apply_exif_orientation)


read = read_insp


def read_preview(
    path: Union[str, Path],
    *,
    mode: Optional[str] = "RGB",
    apply_exif_orientation: bool = False,
) -> np.ndarray:
    """Read the APP2 preview JPEG as a NumPy array."""

    preview = extract_preview_jpeg(path)
    return _decode_image(
        BytesIO(preview),
        mode=mode,
        apply_exif_orientation=apply_exif_orientation,
    )


def read_metadata(path: Union[str, Path]) -> InspMetadata:
    """Inspect the JPEG structure and best-effort Insta360 trailer metadata."""

    path = Path(path)
    data = path.read_bytes()
    width, height, precision, components = _jpeg_shape(data)
    preview = _preview_jpeg(data)
    preview_size = None
    if preview is not None:
        preview_width, preview_height, _, _ = _jpeg_shape(preview)
        preview_size = (preview_width, preview_height)

    raw_trailer = _trailer_bytes(data)
    return InspMetadata(
        path=path,
        width=width,
        height=height,
        precision=precision,
        components=components,
        preview_size=preview_size,
        preview_bytes=len(preview) if preview is not None else 0,
        trailer_bytes=len(raw_trailer),
        exif=_read_exif(path),
        trailer=parse_trailer(raw_trailer) if raw_trailer else None,
    )


def primary_jpeg_bytes(path: Union[str, Path]) -> bytes:
    """Return the primary JPEG stream, excluding the post-EOI trailer."""

    data = Path(path).read_bytes()
    eoi = data.rfind(JPEG_EOI)
    if eoi < 0:
        raise InspFormatError("missing JPEG EOI marker")
    return data[: eoi + len(JPEG_EOI)]


def trailer_bytes(path: Union[str, Path]) -> bytes:
    """Return the bytes after the final JPEG EOI marker."""

    return _trailer_bytes(Path(path).read_bytes())


def extract_preview_jpeg(path: Union[str, Path]) -> bytes:
    """Return the JPEG preview stored across APP2 segments."""

    data = Path(path).read_bytes()
    preview = _preview_jpeg(data)
    if preview is None:
        raise InspFormatError("missing APP2 preview JPEG")
    return preview


def parse_trailer(raw: bytes) -> InspTrailer:
    """Best-effort decoder for the protobuf-like Insta360 trailer."""

    start, rows = _find_trailer_message(raw)
    fields: dict[str, Any] = {}
    decoded_until = start

    for field, wire_type, value, _, end in rows:
        decoded_until = end
        name = TRAILER_FIELD_NAMES.get(field, f"field_{field}")
        fields[name] = _decode_trailer_value(value, wire_type)

    return InspTrailer(
        raw=raw,
        prefix=raw[:start],
        fields=fields,
        decoded_until=decoded_until,
    )


def _decode_image(
    source: Union[str, Path, BinaryIO],
    *,
    mode: Optional[str],
    apply_exif_orientation: bool,
) -> np.ndarray:
    with Image.open(source) as image:
        if apply_exif_orientation:
            image = ImageOps.exif_transpose(image)
        if mode is not None:
            image = image.convert(mode)
        return np.array(image)


def _iter_header_segments(data: bytes) -> Iterable[JPEGSegment]:
    if not data.startswith(JPEG_SOI):
        raise InspFormatError("missing JPEG SOI marker")

    offset = len(JPEG_SOI)
    while offset < len(data):
        if data[offset] != 0xFF:
            raise InspFormatError(f"expected JPEG marker at byte {offset}")

        marker_offset = offset
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            raise InspFormatError("truncated JPEG marker")

        marker = data[offset]
        offset += 1

        if marker == SOS:
            length = _segment_length(data, offset)
            payload = data[offset + 2 : offset + length]
            yield JPEGSegment(marker=marker, offset=marker_offset, payload=payload)
            return

        if marker == 0xD9:
            yield JPEGSegment(marker=marker, offset=marker_offset, payload=b"")
            return

        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            yield JPEGSegment(marker=marker, offset=marker_offset, payload=b"")
            continue

        length = _segment_length(data, offset)
        payload = data[offset + 2 : offset + length]
        yield JPEGSegment(marker=marker, offset=marker_offset, payload=payload)
        offset += length

    raise InspFormatError("truncated JPEG header")


def _segment_length(data: bytes, offset: int) -> int:
    if offset + 2 > len(data):
        raise InspFormatError("truncated JPEG segment length")
    length = int.from_bytes(data[offset : offset + 2], "big")
    if length < 2:
        raise InspFormatError("invalid JPEG segment length")
    if offset + length > len(data):
        raise InspFormatError("truncated JPEG segment payload")
    return length


def _jpeg_shape(data: bytes) -> tuple[int, int, int, int]:
    for segment in _iter_header_segments(data):
        if segment.marker in SOF_MARKERS:
            payload = segment.payload
            if len(payload) < 6:
                raise InspFormatError("truncated JPEG SOF segment")
            precision = payload[0]
            height = int.from_bytes(payload[1:3], "big")
            width = int.from_bytes(payload[3:5], "big")
            components = payload[5]
            return width, height, precision, components

        if segment.marker == SOS:
            break

    raise InspFormatError("missing JPEG SOF segment")


def _preview_jpeg(data: bytes) -> Optional[bytes]:
    chunks = [
        segment.payload
        for segment in _iter_header_segments(data)
        if segment.marker == APP2
    ]
    if not chunks:
        return None

    preview = b"".join(chunks)
    if not preview.startswith(JPEG_SOI) or not preview.endswith(JPEG_EOI):
        raise InspFormatError("APP2 payloads do not form a JPEG preview")
    return preview


def _trailer_bytes(data: bytes) -> bytes:
    eoi = data.rfind(JPEG_EOI)
    if eoi < 0:
        raise InspFormatError("missing JPEG EOI marker")
    return data[eoi + len(JPEG_EOI) :]


def _read_exif(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        exif = image.getexif()
        out: dict[str, Any] = {}
        for tag, value in exif.items():
            name = ExifTags.TAGS.get(tag, str(tag))
            out[name] = value
        return out


def _find_trailer_message(raw: bytes) -> tuple[int, list[tuple[int, int, Any, int, int]]]:
    best_start = 0
    best_rows: list[tuple[int, int, Any, int, int]] = []
    best_score = (-1, -1)

    for start in range(min(len(raw), 32)):
        rows = _parse_protobuf_prefix(raw, start)
        if not rows:
            continue

        names = {field for field, _, _, _, _ in rows}
        text_fields = sum(
            1
            for field, wire_type, value, _, _ in rows
            if field in {1, 2, 3, 5} and wire_type == 2 and _looks_like_text(value)
        )
        score = (text_fields + len(names & {1, 2, 3, 5, 19, 26, 27}), len(rows))
        if score > best_score:
            best_start = start
            best_rows = rows
            best_score = score

    return best_start, best_rows


def _parse_protobuf_prefix(
    raw: bytes,
    start: int,
) -> list[tuple[int, int, Any, int, int]]:
    offset = start
    rows: list[tuple[int, int, Any, int, int]] = []

    while offset < len(raw):
        field_start = offset
        try:
            key, offset = _read_varint(raw, offset)
            field = key >> 3
            wire_type = key & 0x07

            if field == 0:
                break

            if wire_type == 0:
                value, offset = _read_varint(raw, offset)
            elif wire_type == 1:
                if offset + 8 > len(raw):
                    break
                value = raw[offset : offset + 8]
                offset += 8
            elif wire_type == 2:
                length, offset = _read_varint(raw, offset)
                if offset + length > len(raw):
                    break
                value = raw[offset : offset + length]
                offset += length
            elif wire_type == 5:
                if offset + 4 > len(raw):
                    break
                value = raw[offset : offset + 4]
                offset += 4
            else:
                break
        except InspFormatError:
            break

        rows.append((field, wire_type, value, field_start, offset))

    return rows


def _read_varint(raw: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(raw):
        byte = raw[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift > 70:
            break
    raise InspFormatError("invalid varint")


def _decode_trailer_value(value: Any, wire_type: int) -> Any:
    if wire_type == 0:
        return value
    if wire_type == 1 and isinstance(value, bytes):
        return int.from_bytes(value, "little")
    if wire_type == 2 and isinstance(value, bytes):
        if _looks_like_text(value):
            return value.decode("utf-8")
        nested = _parse_protobuf_prefix(value, 0)
        if nested and nested[-1][-1] == len(value):
            return {
                TRAILER_FIELD_NAMES.get(field, f"field_{field}"): _decode_trailer_value(
                    nested_value, nested_wire_type
                )
                for field, nested_wire_type, nested_value, _, _ in nested
            }
        return value
    if wire_type == 5 and isinstance(value, bytes):
        return int.from_bytes(value, "little")
    return value


def _looks_like_text(value: bytes) -> bool:
    if not value:
        return False
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return all(32 <= ord(char) < 127 for char in text)
