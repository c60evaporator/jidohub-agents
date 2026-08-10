"""画像デコーダの登録ポリシーの検証（CLAUDE.md 2.8）。"""

from __future__ import annotations

import io
from typing import Iterator

import numpy as np
import pytest
from jidohub.core.schemas import (
    EncodedPixels,
    ImageDecodeError,
    get_image_decoder,
    register_image_decoder,
)

from jidohub.agents.decoders import pillow_decoder, register_default_decoder


@pytest.fixture(autouse=True)
def restore_decoder() -> Iterator[None]:
    """各テストの前後で登録済みデコーダを保存・復元する。"""
    saved = get_image_decoder()
    try:
        yield
    finally:
        register_image_decoder(saved)


def _jpeg_bytes(width: int, height: int) -> bytes:
    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", (width, height), (10, 20, 30)).save(buf, format="JPEG")
    return buf.getvalue()


def test_default_decoder_is_registered_on_import() -> None:
    # import jidohub.agents で登録済み。
    assert get_image_decoder() is not None


def test_does_not_override_existing() -> None:
    sentinel = lambda enc: np.zeros((enc.height, enc.width, 3), dtype=np.uint8)  # noqa: E731
    register_image_decoder(sentinel)
    assert register_default_decoder() is False
    assert get_image_decoder() is sentinel


def test_force_overrides_existing() -> None:
    sentinel = lambda enc: np.zeros((enc.height, enc.width, 3), dtype=np.uint8)  # noqa: E731
    register_image_decoder(sentinel)
    assert register_default_decoder(force=True) is True
    assert get_image_decoder() is not sentinel


def test_registers_when_none() -> None:
    register_image_decoder(None)
    assert register_default_decoder() is True
    assert get_image_decoder() is not None


def test_pillow_decoder_roundtrip() -> None:
    enc = EncodedPixels.from_bytes(_jpeg_bytes(4, 3), "jpeg", height=3, width=4)
    array = pillow_decoder(enc)
    assert array.shape == (3, 4, 3)
    assert array.dtype == np.uint8


def test_unregistered_raises_image_decode_error() -> None:
    # 未登録時は core の ImageDecodeError（ImportError ではない）が出ることを保つ。
    from jidohub.core.schemas import Image

    register_image_decoder(None)
    enc = EncodedPixels.from_bytes(_jpeg_bytes(4, 3), "jpeg", height=3, width=4)
    image = Image(encoded=enc)
    with pytest.raises(ImageDecodeError):
        _ = image.array
