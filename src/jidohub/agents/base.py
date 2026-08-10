"""Agent の契約。

**このモジュールに具体的なモデル実装を書かない**（CLAUDE.md 1 章）。
契約が変わると全 Agent に影響するため、変更は影響範囲を明示して提案すること。

構成
    - :class:`BaseAgent` — 全 Agent の基底。``from_pretrained`` のテンプレートを提供する
    - :class:`StreamingMixin` — 状態を持つ Agent の能力。**タスク横断**であり
      tracking 専用ではない（``sensing_to_planning`` の時系列モデルも使う）
    - タスク別の抽象クラス — ``predict`` のシグネチャで**出力型を固定**する
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jidohub.core.config import AgentConfig
from jidohub.core.schemas import (
    Classification2DOutput,
    Detection2DOutput,
    Detection3DOutput,
    E2EOutput,
    ImageSample,
    InstanceSegmentation2DOutput,
    MapOutput,
    Sample,
)
from jidohub.core.tasks import TaskType

from jidohub.agents.exceptions import (
    AgentResolutionError,
    IsolationViolationError,
    StateNotInitializedError,
)

if TYPE_CHECKING:
    from jidohub.core.hub import AgentReference

__all__ = [
    "BaseAgent",
    "StreamingMixin",
    "Detection3DAgent",
    "MapConstructionAgent",
    "E2EAgent",
    "Detection2DAgent",
    "InstanceSegmentation2DAgent",
    "Classification2DAgent",
]


class BaseAgent(ABC):
    """全 Agent の基底クラス。

    Agent 作者が実装するのは :meth:`_from_config` / :meth:`load_weights` /
    :meth:`predict` の 3 つ。:meth:`from_pretrained` は基底が提供する
    テンプレートメソッドであり、**override してはならない**（CLAUDE.md 2.2）。
    共通の検証（隔離要件、スキーマ互換、チェックサム）が飛ばされるため。

    Attributes:
        config: ``agent_config.json`` の内容。
        repo_path: ローカルに展開された Agent リポジトリのパス。
        device: モデルを載せるデバイス（``"cuda:0"`` / ``"cpu"`` など）。

    Note:
        ジェネリック（``Generic[InputT, OutputT]``）にしていないのは、
        ``AutoAgent.from_pretrained()`` の戻り値が実行時にしか決まらず
        型引数が ``Any`` に潰れるため（CLAUDE.md 2.1）。入出力の型は
        タスク別の抽象クラスが ``predict`` のシグネチャで固定する。
    """

    config: AgentConfig
    repo_path: Path
    device: str

    #: このクラスが担当するタスク。サブクラスが必ず設定する。
    #: ``AutoAgent`` が config の ``task`` と突き合わせて検証する。
    task: TaskType

    @classmethod
    def from_pretrained(
        cls,
        reference: str | Path | AgentReference,
        revision: str | None = None,
        device: str = "cpu",
        **kwargs: Any,
    ) -> BaseAgent:
        """Agent を取得・構築して返す。**override しないこと。**

        手順
            1. ``HubClient.snapshot()`` でリポジトリをローカルに用意（core が担当）
            2. ``load_agent_config()`` で config を読む（core が担当）
            3. **隔離要件の検証**（``remote_code`` を inprocess で動かさない。2.4）
            4. ``cls._from_config()`` でモデルと Processor を構築
            5. ``verify_weights()`` の後、``load_weights()`` で重みを読み込む

        Args:
            reference: Agent の参照（``"acme/Foo@v1"`` / ローカルパス / S3 URI）。
            revision: リビジョン。``reference`` に ``@`` 表記がある場合は指定しない。
            device: モデルを載せるデバイス。``_from_config`` に引き渡される。
                Agent 作者が環境変数を直接読む実装をしないこと（2.2）。
            **kwargs: ``_from_config`` に渡される追加設定。

        Note:
            Transformers の ``device_map``（モデルを複数 GPU へ分割配置する）とは
            異なり、単一デバイスの指定のみを扱う。自動運転モデルは単一 GPU に
            収まるため、sharding の複雑さを持ち込まない。必要になれば
            引数追加という非破壊変更で対応できる。
        """
        from jidohub.core.hub import HubClient

        # 1. 取得 + config 読込（core が担当）。ローカル参照はコピーせずそのパスが返る。
        client = HubClient()
        config, repo_path = client.load_config(reference, revision)

        # 2. 宣言（config.task）と実装（cls.task）の食い違いを早期に検出する。
        if config.task != cls.task:
            raise AgentResolutionError(
                f"task mismatch: {cls.__name__} implements {cls.task.value!r} "
                f"but agent_config.json declares {config.task.value!r}"
            )

        # 3. 隔離要件の検証（未審査コードを inprocess で動かさない。CLAUDE.md 2.4）。
        #    現段階は Runner が inprocess しかないため、required は明示的なエラーにする
        #    （黙って inprocess で動かさない）。AutoAgent 側でも二重に検証する。
        if config.runtime.isolation == "required":
            raise IsolationViolationError(
                f"{config.agent_id} declares runtime.isolation='required' and cannot run "
                "in-process; a docker runner is required (not yet implemented)."
            )

        # 4. モデルと Processor を構築（Agent 作者の _from_config）。重みはまだ読まない。
        agent = cls._from_config(config, repo_path, device, **kwargs)
        agent.config = config
        agent.repo_path = repo_path
        agent.device = device

        # 5. チェックサム検証の後、各重みを読み込む。重みを持たない Agent
        #    （ルールベース等）では config.weights が空なので load_weights を呼ばない。
        if config.weights:
            client.verify_weights(config, repo_path)
            for weight in config.weights:
                agent.load_weights(repo_path / weight.path)

        return agent

    @classmethod
    @abstractmethod
    def _from_config(
        cls,
        config: AgentConfig,
        repo_path: Path,
        device: str,
        **kwargs: Any,
    ) -> BaseAgent:
        """config からモデルと Processor を構築する（**Agent 作者が実装**）。

        重みの読み込みは行わない（:meth:`load_weights` の責務）。
        モデル固有のハイパーパラメータは ``config.params`` から読む。
        """

    @abstractmethod
    def load_weights(self, path: Path) -> None:
        """重みを読み込む（**Agent 作者が実装**）。

        Args:
            path: 重みファイルの絶対パス。``config.weights`` の各要素に対して
                呼ばれる。チェックサムの検証は :meth:`from_pretrained` が済ませている。
        """

    @abstractmethod
    def predict(self, input: Any) -> Any:
        """推論を実行する（**Agent 作者が実装**）。

        **入力は単一引数**とする。``pack(input)`` → RPC → ``unpack`` → ``predict``
        という経路が引数の増加に耐えないため（CLAUDE.md 2.1）。複合入力が必要な
        タスクは ``<TaskName>Input`` 型を **core に**定義して 1 引数にまとめる。

        出力型はタスク別の抽象クラスが固定する。
        """


class StreamingMixin:
    """状態を持ち、フレーム単位で推論する Agent の能力。

    **tracking 専用ではない。** ``sensing_to_planning`` の時系列モデル（UniAD 等）も
    同じ機構を使うため、タスク横断の能力として定義する（CLAUDE.md 2.3）。

    継承順序
        ``class XAgent(StreamingMixin, BaseAgent)`` の順で継承すること。
        逆順（``BaseAgent, StreamingMixin``）だと MRO 上 :class:`BaseAgent` が
        先に来るため、本 Mixin の ``predict`` が abstract を実装したことにならず、
        **インスタンス化時に「predict が未実装」という TypeError になる**。

        落ちること自体は正しいが、``reset`` / ``step`` / ``_aggregate`` を
        実装済みの作者にはエラー文から原因が読み取れない。
        :func:`jidohub.agents.testing.check_mro_order` が明示的なメッセージを出す。

    役割分担
        - jidohub-interfaces（実車・シミュレーション）→ :meth:`reset` と :meth:`step`
        - 評価ハーネス・jidohub-server（オフライン一括）→ :meth:`predict`

    **1 つの実装からオンライン推論とオフライン評価の両方が出る**ことが設計の要点。
    """

    _initialized: bool = False

    def reset(self) -> None:
        """内部状態を初期化する（**Agent 作者が実装**、``super().reset()`` を呼ぶこと）。

        何度呼んでもよい（冪等）。track_id の採番もここでリセットする。

        Note:
            ``track_id`` の一意性は **``reset()`` から次の ``reset()`` までの
            セッション内**に限る。セッションをまたいで同じ整数が別の物体を
            指してよい（Adapter の ``instance_token`` 写像とはスコープが異なる）。
        """
        self._initialized = True

    def step(self, input: Any) -> Any:
        """1 フレーム分の入力に対する出力を返す（**Agent 作者が実装**）。

        実装は先頭で :meth:`_check_initialized` を呼ぶこと。
        """
        raise NotImplementedError

    def predict(self, inputs: list) -> Any:
        """系列全体をまとめて処理する（**基底の既定実装。override しないこと**）。

        :meth:`reset` してから :meth:`step` をループし、各フレームの出力を集約する。

        **不変条件**: 本メソッドの結果は、``reset()`` してから ``step()`` を手動で
        ループした結果と**一致しなければならない**。一致しない場合、既定実装を
        上書きしているか、``step()`` が入力以外の状態に依存している。
        共通テストで機械的に検証する（CLAUDE.md 3.1）。
        """
        self.reset()
        frames = [self.step(item) for item in inputs]
        return self._aggregate(frames)

    @abstractmethod
    def _aggregate(self, frames: list) -> Any:
        """:meth:`step` の出力列を系列の出力型にまとめる（**Agent 作者が実装**）。

        系列の出力型は、単体タスクの出力型を時刻順に保持する薄いラッパとする
        （中間出力を単体タスクの出力型で表す既存方針と同じ）。これにより
        1 フレーム取り出せば既存の可視化コードがそのまま使える。
        """

    def _check_initialized(self) -> None:
        """``reset()`` 済みかを検証する。:meth:`step` の実装が先頭で呼ぶ。

        未 reset を許すと**前シーンの内部状態が漏れる**。track_id が引き継がれ、
        評価結果を静かに汚染するため、明示的なエラーにする（CLAUDE.md 2.3）。
        """
        if not self._initialized:
            raise StateNotInitializedError(
                f"{type(self).__name__}.reset() must be called before step(). "
                "Streaming agents keep internal state across frames."
            )


# --- タスク別の抽象クラス -----------------------------------------------------
#
# predict のシグネチャで**出力型を固定**する。TaskType と出力型の 1 対 1 対応を
# 型の上で表現するのが目的であり、実装は持たない。
#
# ストリーミングが必要なタスク（tracking 等）は、これらと StreamingMixin を
# 併用する。複合入力型（Tracking3DInput 等）は該当 Agent の実装時に core へ
# 定義するため、ここではまだ用意しない（CLAUDE.md 4 章）。


class Detection3DAgent(BaseAgent):
    """3D 物体検出（``object_detection_3d``）。"""

    task = TaskType.OBJECT_DETECTION_3D

    @abstractmethod
    def predict(self, input: Sample) -> Detection3DOutput: ...


class MapConstructionAgent(BaseAgent):
    """オンライン HD マップ構築（``map_construction``）。"""

    task = TaskType.MAP_CONSTRUCTION

    @abstractmethod
    def predict(self, input: Sample) -> MapOutput: ...


class E2EAgent(BaseAgent):
    """End-to-end 自動運転（``sensing_to_planning``）。

    時系列モデル（UniAD 等）は :class:`StreamingMixin` を併用する::

        class UniADAgent(StreamingMixin, E2EAgent): ...
    """

    task = TaskType.SENSING_TO_PLANNING

    @abstractmethod
    def predict(self, input: Sample) -> E2EOutput: ...


class Detection2DAgent(BaseAgent):
    """2D 物体検出（``object_detection_2d``）。

    オープン語彙検出（Grounding DINO 等）も同じタスク。テキストプロンプトは
    ``input.prompt.texts`` で受け、要否は ``config.prompt`` が宣言する。

    **出力座標は入力 ``Image`` の現サイズ基準**でなければならない。
    モデル入力用に resize した場合、戻すのは Agent の責務（CLAUDE.md 2.6）。
    """

    task = TaskType.OBJECT_DETECTION_2D

    @abstractmethod
    def predict(self, input: ImageSample) -> Detection2DOutput: ...


class InstanceSegmentation2DAgent(BaseAgent):
    """インスタンスセグメンテーション（``instance_segmentation_2d``）。

    プロンプタブル分割（SAM2 / SAM3 等）も同じタスク。

    **マスクは二値化の前に ``F.interpolate`` で入力解像度へ戻すこと。**
    二値化してから最近傍で拡大すると境界が劣化する（CLAUDE.md 2.6）。
    """

    task = TaskType.INSTANCE_SEGMENTATION_2D

    @abstractmethod
    def predict(self, input: ImageSample) -> InstanceSegmentation2DOutput: ...


class Classification2DAgent(BaseAgent):
    """画像分類（``image_classification_2d``）。

    ゼロショット分類（SigLIP2 等）も同じタスク。ラベル候補は
    ``input.prompt.texts`` で受ける。
    """

    task = TaskType.IMAGE_CLASSIFICATION_2D

    @abstractmethod
    def predict(self, input: ImageSample) -> Classification2DOutput: ...
