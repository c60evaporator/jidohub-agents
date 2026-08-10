"""jidohub-agents: Agent をロードして実行する Python API。

core にのみ依存する（星形依存）。**torch はここで初めて必須依存になる**。

import 時に画像デコーダを登録する。**既に登録済みの場合は上書きしない**
（利用者が nvJPEG 等を先に登録している可能性があるため。CLAUDE.md 2.8）。
"""

from __future__ import annotations

from jidohub.agents.auto import AutoAgent
from jidohub.agents.base import (
    BaseAgent,
    Classification2DAgent,
    Detection2DAgent,
    Detection3DAgent,
    E2EAgent,
    InstanceSegmentation2DAgent,
    MapConstructionAgent,
    StreamingMixin,
)
from jidohub.agents.decoders import register_default_decoder
from jidohub.agents.exceptions import (
    AgentError,
    AgentResolutionError,
    IsolationViolationError,
    StateNotInitializedError,
)
from jidohub.agents.processing import PreprocessResult
from jidohub.agents.registry import NATIVE_AGENTS

# import 時に既定デコーダを登録する（jidohub.datasets と同じ挙動）。
# **既に登録済みの場合は上書きしない**（利用者が nvJPEG 等を先に登録している可能性。CLAUDE.md 2.8）。
register_default_decoder()

__all__ = [
    "AutoAgent",
    "BaseAgent",
    "StreamingMixin",
    "Detection3DAgent",
    "MapConstructionAgent",
    "E2EAgent",
    "Detection2DAgent",
    "InstanceSegmentation2DAgent",
    "Classification2DAgent",
    "PreprocessResult",
    "NATIVE_AGENTS",
    "register_default_decoder",
    "AgentError",
    "AgentResolutionError",
    "IsolationViolationError",
    "StateNotInitializedError",
]
