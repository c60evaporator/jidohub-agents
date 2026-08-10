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
"""

from __future__ import annotations

__all__ = ["register_default_decoder"]


def register_default_decoder(force: bool = False) -> bool:
    """利用可能な最速のデコーダを登録する。

    Args:
        force: ``True`` なら既存の登録を上書きする。

    Returns:
        登録した場合は ``True``、既存の登録を尊重してスキップした場合は ``False``。
    """
    raise NotImplementedError
