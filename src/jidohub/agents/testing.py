"""Agent の共通テストスイート。

``BaseAgent`` を実装する Agent が満たすべき性質を**再利用可能な形**で提供する。
ネイティブ実装にも、外部の Agent 作者にも同じ検証を適用できるようにするため
（CLAUDE.md 3.1）。

Example:
    >>> from jidohub.agents.testing import check_agent_contract
    >>> check_agent_contract(agent, sample_input)
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "check_agent_contract",
    "check_streaming_contract",
    "check_mro_order",
    "check_output_type",
    "check_2d_output_frame",
]


def check_agent_contract(agent: Any, sample_input: Any) -> None:
    """Agent が契約を満たすことを一括で検証する。"""
    raise NotImplementedError


def check_mro_order(agent_class: type) -> None:
    """``StreamingMixin`` が ``BaseAgent`` より前に来ることを検証する。

    逆順だと Mixin の ``predict`` が abstract を実装したことにならず、
    **インスタンス化時に「predict が未実装」という TypeError** になる。
    落ちること自体は正しいが、``reset`` / ``step`` / ``_aggregate`` を実装済みの
    作者にはエラー文から原因が読み取れないため、明示的なメッセージを出す。
    """
    raise NotImplementedError


def check_output_type(agent: Any, sample_input: Any) -> None:
    """``predict`` の出力型が ``TaskType`` の宣言と一致することを検証する。"""
    raise NotImplementedError


def check_streaming_contract(agent: Any, sample_inputs: list) -> None:
    """ストリーミング Agent の不変条件を検証する。

    - ``predict(inputs)`` の結果が ``reset()`` + ``step()`` ループと**一致**する
    - 未 ``reset()`` での ``step()`` が ``StateNotInitializedError`` になる
    - ``reset()`` が冪等である

    一致しない場合、既定 ``predict()`` を上書きしているか、``step()`` が
    入力以外の状態に依存している。
    """
    raise NotImplementedError


def check_2d_output_frame(agent: Any, image_sample: Any) -> None:
    """2D Agent の出力座標が**入力 ``Image`` の現サイズ基準**であることを検証する。

    モデル入力用に resize したまま戻し忘れると、座標が数倍ずれる。
    例外は出ないため、入力サイズとの整合を機械的に確認する（CLAUDE.md 2.6）。
    """
    raise NotImplementedError
