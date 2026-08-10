"""jidohub-agents の例外型。

core の例外（``ConfigValidationError`` / ``SerializationError`` / ``ImageDecodeError``）は
そのまま送出する。ここで定義するのは agents 固有の失敗のみ。
"""

from __future__ import annotations

__all__ = [
    "AgentError",
    "AgentResolutionError",
    "IsolationViolationError",
    "StateNotInitializedError",
]


class AgentError(Exception):
    """jidohub-agents の基底例外。"""


class AgentResolutionError(AgentError):
    """``agent_config.json`` から Agent クラスを解決できなかった場合。

    ``native_class`` がレジストリに無い、``auto_map`` の指すクラスが存在しない、
    タスクと出力型の対応が取れない、などが該当する。
    """


class IsolationViolationError(AgentError):
    """隔離要件に反する実行が要求された場合。

    未審査コード（``implementation.type == "remote_code"``）を inprocess で
    実行しようとした場合など。**セキュリティ境界の違反であり、回避手段を
    提供しないこと**（CLAUDE.md 2.4）。
    """


class StateNotInitializedError(AgentError):
    """``reset()`` を呼ばずに ``step()`` が呼ばれた場合。

    前シーンの内部状態が漏れると track_id が引き継がれ、評価結果を静かに
    汚染するため、明示的なエラーにする（CLAUDE.md 2.3）。
    """
