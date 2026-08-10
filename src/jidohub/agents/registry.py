"""ネイティブ実装のレジストリ。

``agent_config.json`` の ``implementation.native_class`` をここから解決する。

**明示的な辞書で列挙する。** エントリポイント経由の暗黙登録にしないのは、
「どのクラスが審査済みか」が一覧で読めることを優先するため（CLAUDE.md 2.4）。
セキュリティ設計の一部であり、利便性のために暗黙化しないこと。

ネイティブ実装は**純 PyTorch に保つ**（CLAUDE.md 2.5）。``mmcv`` / ``spconv`` 等の
ビルドが不安定な依存を持つモデルは、``remote_code`` + docker runner の側で扱う。
"""

from __future__ import annotations

from jidohub.agents.base import BaseAgent
from jidohub.agents.exceptions import AgentResolutionError

__all__ = ["NATIVE_AGENTS", "resolve_native_agent"]

NATIVE_AGENTS: dict[str, type[BaseAgent]] = {
    # "CenterPointAgent": CenterPointAgent,   # native/centerpoint で実装後に登録する
}
"""``native_class`` 名 → Agent クラス。**審査済み実装の一覧**でもある。"""


def resolve_native_agent(native_class: str) -> type[BaseAgent]:
    """``native_class`` 名から Agent クラスを引く。

    Raises:
        AgentResolutionError: 未登録の場合。利用可能な名前を併記して、
            typo と「未実装」を利用者が区別できるようにする。
    """
    agent_class = NATIVE_AGENTS.get(native_class)
    if agent_class is None:
        available = ", ".join(sorted(NATIVE_AGENTS)) or "(none registered)"
        raise AgentResolutionError(
            f"native_class {native_class!r} is not registered. Available: {available}"
        )
    return agent_class
