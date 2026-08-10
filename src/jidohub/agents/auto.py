"""``AutoAgent``: 参照から Agent を解決して構築する。

責務の境界
    **取得・config の検証・スキーマ互換の判定は core**（``HubClient`` /
    ``load_agent_config``）。agents が担うのは**クラス解決・重みロード・実行**である。

二層構造（CLAUDE.md 2.4）
    ============== ============================== ==============
    implementation 解決方法                        実行
    ============== ============================== ==============
    ``native``     :mod:`jidohub.agents.registry`  inprocess 可
    ``remote_code`` ``auto_map`` から動的ロード     **隔離必須**
    ============== ============================== ==============
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jidohub.agents.base import BaseAgent
    from jidohub.core.config import AgentConfig
    from jidohub.core.hub import AgentReference

__all__ = ["AutoAgent"]


class AutoAgent:
    """``agent_config.json`` から適切な Agent クラスを解決して構築する。

    Example:
        >>> agent = AutoAgent.from_pretrained("acme/UniAD-tiny@v1.2")
        >>> output = agent.predict(sample)
    """

    @classmethod
    def from_pretrained(
        cls,
        reference: str | Path | AgentReference,
        revision: str | None = None,
        device: str = "cpu",
        runner: str = "auto",
        transport: str = "auto",
        **kwargs: Any,
    ) -> BaseAgent:
        """Agent を取得・解決・構築して返す。

        Args:
            reference: Agent の参照。
            revision: リビジョン。
            device: モデルを載せるデバイス。
            runner: ``"auto"`` / ``"inprocess"`` / ``"docker"``。
                ``"auto"`` は ``config.runtime.isolation`` から決める
                （``required`` なら docker、それ以外は inprocess）。
                **実行環境の性質であり Agent の性質ではない**ため、
                ``agent_config.json`` ではなくここで指定する。
            transport: ``"auto"`` / ``"stream"`` / ``"shm"``。docker runner でのみ有効。
                **``"auto"`` は ``"shm"`` を選ばない。** 共有メモリの利得は
                「画素が生で供給され、生産側が共有バッファへ直接書ける」ときにしか
                生じず、Runner 側からは判定できないため。
            **kwargs: Agent の ``_from_config`` に渡される。

        Raises:
            AgentResolutionError: クラスを解決できない場合。
            IsolationViolationError: 隔離要件に反する組み合わせが指定された場合。
        """
        raise NotImplementedError

    @classmethod
    def _resolve_class(cls, config: AgentConfig, repo_path: Path) -> type[BaseAgent]:
        """config から Agent クラスを解決する。

        ``native`` はレジストリから、``remote_code`` は ``auto_map`` のパスから
        動的ロードする。解決後、**クラスの ``task`` が config の ``task`` と
        一致すること**を検証する（宣言と実装の食い違いを早期に検出するため）。
        """
        raise NotImplementedError

    @classmethod
    def _check_isolation(cls, config: AgentConfig, runner: str, transport: str) -> None:
        """隔離要件を検証する。**セキュリティ境界であり回避手段を提供しないこと。**

        core の config 検証で ``remote_code`` → ``isolation: required`` は
        強制されているが、**agents 側でも二重に検証する**。config を経由しない
        経路（クラスを直接構築する等）でも破られないようにするため（CLAUDE.md 2.4）。

        - ``isolation == "required"`` の Agent を ``runner="inprocess"`` で実行しない
        - ``remote_code`` の Agent に ``transport="shm"`` を許可しない
          （IPC 名前空間の共有を要し、隔離が弱まるため）
        """
        raise NotImplementedError
