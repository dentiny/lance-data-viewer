import base64

import pyarrow as pa
import pytest

from serialize_value import detect_media_type, serialize_value


@pytest.mark.parametrize(
    ("payload", "media_type", "mime_type"),
    [
        (b"\x89PNG\r\n\x1a\npayload", "image", "image/png"),
        (b"\xff\xd8\xff\xe0payload", "image", "image/jpeg"),
        (b"RIFF\x00\x00\x00\x00WAVEpayload", "audio", "audio/wav"),
        (b"ID3\x04\x00\x00payload", "audio", "audio/mpeg"),
        (b"\x00\x00\x00\x18ftypisompayload", "video", "video/mp4"),
        (b"\x1aE\xdf\xa3payload", "video", "video/webm"),
    ],
)
def test_detect_media_type(payload, media_type, mime_type):
    assert detect_media_type(payload) == (media_type, mime_type)


def test_media_binary_serialization():
    payload = b"\x89PNG\r\n\x1a\npayload"
    result = serialize_value(pa.scalar(payload, type=pa.large_binary()))
    assert result == {
        "type": "media",
        "media_type": "image",
        "mime_type": "image/png",
        "size": len(payload),
        "base64": base64.b64encode(payload).decode("ascii"),
    }


def test_non_media_binary_serialization_is_unchanged():
    assert serialize_value(b"hello") == "hello"
    payload = b"\xff\xfe\x01\x02"
    assert serialize_value(payload) == base64.b64encode(payload).decode("ascii")
