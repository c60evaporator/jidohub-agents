"""画像デコーダの登録ポリシー。

``frame.image.array`` を呼ぶにはデコーダの登録が必要だが、core は画像コーデックに
依存しない。**agents は datasets に依存しない**（星形依存）ため、agents 側にも
登録の仕組みが要る（CLAUDE.md 2.8）。

優先順位
    1. **既に登録済みなら上書きしない**（利用者が独自デコーダを登録している場合を尊重）
    2. nvJPEG（利用可能なら）— デコード結果が最初から GPU 上に載るため、
       コンテナ内推論で効果が大きい
    3. Pillow

判定は ``importlib.util.find_spec`` で行い、未インストール時に ``ImportError`` ではなく
core の ``ImageDecodeError`` が出る状態を保つ（原因が利用者に伝わるようにするため）。

Note:
    nvJPEG 経路（:func:`torchvision_decoder`）は、core のデコーダ契約が
    「**CPU 上の numpy 配列を返す**」ことを求めるため、GPU 常駐の利得はこの層では
    実現されない（``.cpu().numpy()`` で CPU へ戻す）。共有メモリ転送を伴う
    コンテナ内 RPC 経路（CLAUDE.md 4 章、次段階）で活きる将来最適化として
    優先順位のみ用意する。JPEG 以外（PNG / WebP）は Pillow にフォールバックする。
"""

from __future__ import annotations

import importlib.util
import io

import numpy as np
from jidohub.core.schemas import (
    EncodedPixels,
    ImageFormat,
    get_image_decoder,
    register_image_decoder,
)

__all__ = ["pillow_decoder", "torchvision_decoder", "register_default_decoder"]


def pillow_decoder(encoded: EncodedPixels) -> np.ndarray:
    """Pillow で符号化画像をデコードする。

    :data:`~jidohub.core.schemas.ImageDecoder` の契約通り、
    shape ``(H, W, 3)`` の ``np.uint8`` を **RGB 順**で返す。
    グレースケールや RGBA の画像も RGB に変換する
    （契約が常に 3 チャンネル RGB であるため）。
    """
    from PIL import Image  # 遅延 import（未インストール時は register 側で弾く）。

    with Image.open(io.BytesIO(encoded.to_bytes())) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def torchvision_decoder(encoded: EncodedPixels) -> np.ndarray:
    """torchvision（nvJPEG）で符号化画像をデコードする。

    JPEG のみ ``torchvision.io.decode_jpeg`` で復号し、それ以外の形式は
    :func:`pillow_decoder` に委譲する。契約通り shape ``(H, W, 3)`` の
    ``np.uint8`` を **RGB 順**で返す。

    Note:
        core の契約が CPU 上の numpy を求めるため、ここでは CPU で復号して
        numpy へ戻す（GPU 常駐の利得はこの層では実現されない。モジュール docstring 参照）。
        ``decode_jpeg`` は ``(C, H, W)`` の RGB uint8 テンソルを返すため、
        ``(H, W, C)`` へ並べ替える。
    """
    if encoded.format is not ImageFormat.JPEG:
        # decode_jpeg は JPEG 専用。PNG / WebP は Pillow にフォールバックする。
        return pillow_decoder(encoded)

    import torch  # 遅延 import。
    from torchvision.io import decode_jpeg

    data = torch.frombuffer(bytearray(encoded.to_bytes()), dtype=torch.uint8)
    tensor = decode_jpeg(data)  # (C, H, W) RGB uint8, CPU。
    array = tensor.permute(1, 2, 0).contiguous().numpy()
    return np.ascontiguousarray(array, dtype=np.uint8)


def register_default_decoder(force: bool = False) -> bool:
    """利用可能な最速のデコーダを登録する。

    優先順位（モジュール docstring 参照）
        既登録を尊重 → nvJPEG（torchvision）→ Pillow。

    判定は :func:`importlib.util.find_spec` で行い、実際の import はデコード時まで
    遅延させる。どの候補も無い環境では登録せず ``False`` を返し、``frame.image.array``
    アクセス時に ``ImportError`` ではなく core の
    :class:`~jidohub.core.schemas.ImageDecodeError`（「デコーダが未登録」）が出る状態を保つ。

    Args:
        force: ``True`` なら既存の登録を上書きする。

    Returns:
        登録した場合は ``True``、既存の登録を尊重してスキップした場合、または
        利用可能なデコーダが無く登録できなかった場合は ``False``。
    """
    if not force and get_image_decoder() is not None:
        return False

    if importlib.util.find_spec("torchvision") is not None:
        register_image_decoder(torchvision_decoder)
        return True

    if importlib.util.find_spec("PIL") is not None:
        register_image_decoder(pillow_decoder)
        return True

    return False
