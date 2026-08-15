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

from jidohub.agents.exceptions import IsolationViolationError
from jidohub.agents.registry import resolve_native_agent

if TYPE_CHECKING:
    from jidohub.core.config import AgentConfig
    from jidohub.core.hub import AgentReference

    from jidohub.agents.base import BaseAgent

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
            NotImplementedError: docker runner / shm transport / remote_code など、
                次段階でのみ実装される経路が要求された場合。
        """
        from jidohub.core.hub import AgentReference, HubClient

        # 参照を 1 度だけ解決する（revision を含む）。取得と、委譲先での記録の両方に使う。
        resolved = (
            reference
            if isinstance(reference, AgentReference)
            else AgentReference.parse(reference, revision)
        )

        # 取得 + config 読込（core が担当）。ローカル参照はコピーせずそのパスが返る。
        client = HubClient()
        config, repo_path = client.load_config(resolved)

        # runner="auto" は config.runtime.isolation から決める。
        # **実行環境の性質であり Agent の性質ではない**ため、ここで解決する。
        effective_runner = cls._resolve_runner(config, runner)

        # 隔離検証（agents 側でも二重に検証する。CLAUDE.md 2.4）。
        cls._check_isolation(config, effective_runner, transport)

        # クラス解決（task 整合の検証を含む）。
        agent_class = cls._resolve_class(config, repo_path)

        # 解決したクラスの from_pretrained へ委譲する（二重実装を避ける）。
        # 取得は済んでいるので**ローカル repo_path** を渡し（再取得なし）、revision を含む
        # 解決済み参照は _resolved_reference で別途伝える（そうしないと revision が失われる）。
        return agent_class.from_pretrained(
            repo_path, device=device, _resolved_reference=resolved, **kwargs
        )

    @staticmethod
    def _resolve_runner(config: AgentConfig, runner: str) -> str:
        """``runner="auto"`` を config から解決する。

        ``isolation == "required"`` なら docker、それ以外は inprocess。
        明示指定（``"inprocess"`` / ``"docker"``）はそのまま返す。
        """
        if runner != "auto":
            return runner
        return "docker" if config.runtime.isolation == "required" else "inprocess"

    @classmethod
    def _resolve_class(cls, config: AgentConfig, repo_path: Path) -> type[BaseAgent]:
        """config から Agent クラスを解決する。

        ``native`` はレジストリから解決する。``remote_code`` の動的ロードは
        docker runner を要するため**次段階**であり、ここでは明示的な
        :class:`NotImplementedError` にする（黙って inprocess にフォールバックしない）。

        解決後、**クラスの ``task`` が config の ``task`` と一致すること**を
        検証する（宣言と実装の食い違いを早期に検出するため）。
        """
        from jidohub.agents.exceptions import AgentResolutionError

        if config.implementation.type == "remote_code":
            raise NotImplementedError(
                "remote_code agents require a docker runner to load unreviewed code "
                "in isolation; this is not yet implemented (native + inprocess only)."
            )

        # native。native_class は Implementation の検証で非 None が保証されている。
        assert config.implementation.native_class is not None
        agent_class = resolve_native_agent(config.implementation.native_class)

        if agent_class.task != config.task:
            raise AgentResolutionError(
                f"task mismatch: {agent_class.__name__} implements "
                f"{agent_class.task.value!r} but agent_config.json declares "
                f"{config.task.value!r}"
            )
        return agent_class

    @classmethod
    def _check_isolation(cls, config: AgentConfig, runner: str, transport: str) -> None:
        """隔離要件を検証する。**セキュリティ境界であり回避手段を提供しないこと。**

        core の config 検証で ``remote_code`` → ``isolation: required`` は
        強制されているが、**agents 側でも二重に検証する**。config を経由しない
        経路（クラスを直接構築する等）でも破られないようにするため（CLAUDE.md 2.4）。

        - ``remote_code`` の Agent に ``transport="shm"`` を許可しない
          （IPC 名前空間の共有を要し、隔離が弱まるため）
        - ``transport="shm"`` / ``runner="docker"`` は次段階でのみ実装される
        - ``isolation == "required"`` の Agent を ``runner="inprocess"`` で実行しない
        """
        # shm は remote_code では隔離を弱めるため拒否する（セキュリティ境界）。
        # それ以外でも共有メモリ転送は docker runner を要し、現段階では未実装。
        if transport == "shm":
            if config.implementation.type == "remote_code":
                raise IsolationViolationError(
                    "transport='shm' is refused for remote_code agents: sharing the IPC "
                    "namespace weakens the isolation that unreviewed code requires."
                )
            raise NotImplementedError(
                "transport='shm' requires a docker runner and is not yet implemented."
            )

        if runner == "docker":
            raise NotImplementedError(
                "docker runner is not yet implemented (native + inprocess only)."
            )

        # ここに到達したら runner は inprocess。未審査コード相当（isolation required）を
        # inprocess で動かさない。黙って inprocess にフォールバックしない。
        if config.runtime.isolation == "required":
            raise IsolationViolationError(
                f"{config.agent_id} declares runtime.isolation='required' and cannot run "
                "in-process; a docker runner is required (not yet implemented)."
            )
