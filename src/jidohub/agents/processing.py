"""前処理と後処理の間で受け渡す変換情報。

なぜ戻り値で渡すのか
    モデル入力用に resize / crop / padding を行った場合、**後処理はその逆変換を
    知る必要がある**（SAM の ``postprocess_masks`` がパディング情報を要求するのと同じ）。

    この情報を Agent のインスタンス変数として持ち回ると、ストリーミング時
    （フレームごとに異なる）やバッチ時（サンプルごとに異なる）に取り違える。
    **Processor の戻り値に含めることで、入力と変換情報が構造的に対応する。**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from jidohub.core.schemas import ImageSource

__all__ = ["PreprocessResult"]


@dataclass
class PreprocessResult:
    """前処理の結果と、後処理に必要な変換情報。

    Attributes:
        tensors: モデルに渡すテンソル群。キーはモデル固有。
        transform: 入力 ``Image`` → モデル入力 の変換。resize していなければ ``None``。
            後処理で出力座標を入力 ``Image`` 基準へ戻す際に使う（CLAUDE.md 2.6）。
        input_size: 入力 ``Image`` の ``(height, width)``[px]。
            マスクを ``F.interpolate`` で戻す際の目標サイズ。
        extras: モデル固有の付加情報（パディング量、有効領域など）。
            **標準スキーマで表現できるものをここに入れないこと。**
    """

    tensors: dict[str, "torch.Tensor"]
    transform: "ImageSource | None" = None
    input_size: tuple[int, int] | None = None
    extras: dict[str, Any] = field(default_factory=dict)
