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

import numpy as np
from jidohub.core.config import AgentConfig
from jidohub.core.schemas import (
    Classification2DOutput,
    CoordinateFrame,
    Detection2DOutput,
    Detection2DSequence,
    Detection3DOutput,
    Detection3DSequence,
    E2EOutput,
    E2ESequence,
    ImageSample,
    InstanceSegmentation2DOutput,
    InstanceSegmentation2DSequence,
    MapOutput,
    Sample,
)
from jidohub.core.tasks import TaskType

from jidohub.agents.exceptions import (
    AgentResolutionError,
    IsolationViolationError,
    StateNotInitializedError,
    StreamingContractError,
    UpstreamInputError,
)

if TYPE_CHECKING:
    from jidohub.core.hub import AgentReference
    from jidohub.core.schemas import Tracking2DInput, Tracking3DInput

__all__ = [
    "BaseAgent",
    "StreamingMixin",
    "Detection3DAgent",
    "MapConstructionAgent",
    "E2EAgent",
    "Detection2DAgent",
    "InstanceSegmentation2DAgent",
    "Classification2DAgent",
    "Tracking3DAgent",
    "Tracking2DAgent",
    "InstanceSegmentationTracking2DAgent",
    "StreamingE2EAgent",
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
        reference: 解決済みの参照（``revision`` を含む）。``AutoAgent`` 経由でも
            委譲先に revision が伝わるようにここに保持する。参照を解決できない
            経路（テストで直接構築する等）では ``None``。

    Note:
        ジェネリック（``Generic[InputT, OutputT]``）にしていないのは、
        ``AutoAgent.from_pretrained()`` の戻り値が実行時にしか決まらず
        型引数が ``Any`` に潰れるため（CLAUDE.md 2.1）。入出力の型は
        タスク別の抽象クラスが ``predict`` のシグネチャで固定する。
    """

    config: AgentConfig
    repo_path: Path
    device: str
    reference: AgentReference | None

    #: このクラスが担当するタスク。サブクラスが必ず設定する。
    #: ``AutoAgent`` が config の ``task`` と突き合わせて検証する。
    task: TaskType

    @classmethod
    def from_pretrained(
        cls,
        reference: str | Path | AgentReference,
        revision: str | None = None,
        device: str = "cpu",
        *,
        _resolved_reference: AgentReference | None = None,
        **kwargs: Any,
    ) -> BaseAgent:
        """Agent を取得・構築して返す。**override しないこと。**

        手順
            1. ``HubClient.snapshot()`` でリポジトリをローカルに用意（core が担当）
            2. ``load_agent_config()`` で config を読む（core が担当）
            3. **隔離要件の検証**（``remote_code`` を inprocess で動かさない。2.4）
            4. ``cls._from_config()`` でモデルと Processor を構築
            5. ``verify_weights()`` の後、``load_weights()`` で重みを読み込む
            6. 解決済み参照（revision を含む）を ``self.reference`` に記録する

        Args:
            reference: Agent の参照（``"acme/Foo@v1"`` / ローカルパス / S3 URI）。
            revision: リビジョン。``reference`` に ``@`` 表記がある場合は指定しない。
            device: モデルを載せるデバイス。``_from_config`` に引き渡される。
                Agent 作者が環境変数を直接読む実装をしないこと（2.2）。
            _resolved_reference: **内部用。** ``AutoAgent`` が解決済みの
                :class:`AgentReference`（revision を含む）を渡す経路。``AutoAgent`` は
                取得のため委譲先にローカル ``repo_path`` を渡すが、そのままでは revision が
                失われるため、記録用の参照をこの引数で別途伝える。直接呼びでは ``None`` で、
                ``from_pretrained`` 自身が ``reference`` を解析して記録する。
            **kwargs: ``_from_config`` に渡される追加設定。

        Note:
            Transformers の ``device_map``（モデルを複数 GPU へ分割配置する）とは
            異なり、単一デバイスの指定のみを扱う。自動運転モデルは単一 GPU に
            収まるため、sharding の複雑さを持ち込まない。必要になれば
            引数追加という非破壊変更で対応できる。
        """
        from jidohub.core.hub import AgentReference, HubClient

        # 1. 取得 + config 読込（core が担当）。ローカル参照はコピーせずそのパスが返る。
        #    **取得は常に reference で行う**（AutoAgent はローカル repo_path を渡すため
        #    再取得は発生しない）。revision の記録は _resolved_reference で別途受ける。
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

        # 6. 解決済み参照（revision を含む）を記録する。AutoAgent が渡した参照を
        #    優先し、無ければ reference を解析する（ローカルパスは revision=None）。
        agent.reference = (
            _resolved_reference
            if _resolved_reference is not None
            else reference
            if isinstance(reference, AgentReference)
            else AgentReference.parse(reference, revision)
        )

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

    単発のタスク別抽象クラスとは併用しない
        ``Detection3DAgent`` 等の**単発**タスク別抽象クラスは ``predict(input: Sample)`` を
        宣言するが、本 Mixin の ``predict`` は **系列（``list``）** を取る。両者は
        シグネチャが非互換であり、合成すると型検査が正しく矛盾を報告する。
        したがって単発クラスを継承せず **``class X(StreamingMixin, BaseAgent)``** とする。
        ストリーミングが必要なタスク（tracking や時系列 E2E）は、系列出力型を固定する
        **専用のタスク別抽象クラス**（:class:`Tracking3DAgent` / :class:`Tracking2DAgent` /
        :class:`InstanceSegmentationTracking2DAgent` / :class:`StreamingE2EAgent`。本ファイル末尾）
        を継承する。これらは ``StreamingMixin`` + ``BaseAgent`` の組であり、``_aggregate`` を
        実装済みなので、**Agent 作者は :meth:`reset` と :meth:`step` だけを書けばよい**。

    メタ情報の収集
        既定 :meth:`predict` は :meth:`_collect_meta` で各入力から ``.timestamp`` と
        ``ego_to_global`` を集め、系列出力型へ引き写す。**系列入力型は ``.timestamp`` を
        公開する**ことを契約とする（streaming_agents.md 3.2）。

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

        入力からメタ情報（``timestamps`` / ``ego_to_global``）を集め、:meth:`reset` して
        から :meth:`step` をループし、各フレームの出力を系列出力型へ集約する。
        検証（メタ情報・上流入力）は**ループの前**に済ませる。100 フレーム推論した後で
        落ちるのを避けるためである（streaming_agents.md 3.2）。

        **不変条件**: 本メソッドの結果は、``reset()`` してから ``step()`` を手動で
        ループした結果と**一致しなければならない**。一致しない場合、既定実装を
        上書きしているか、``step()`` が入力以外の状態に依存している。
        共通テストで機械的に検証する（CLAUDE.md 3.1）。
        """
        timestamps, ego_poses = self._collect_meta(inputs)
        self._check_requirements(inputs)
        self.reset()
        frames = [self.step(item) for item in inputs]
        return self._aggregate(frames, timestamps, ego_poses)

    def _collect_meta(self, inputs: list) -> tuple[np.ndarray, np.ndarray | None]:
        """入力の系列からメタ情報（``timestamps`` / ``ego_to_global``）を収集する。

        **系列入力型が ``.timestamp`` を公開する**ことを契約とする（streaming_agents.md 3.2）。
        入力型ごとに分岐しないよう ``getattr`` で一様に収集する。``Sample`` は
        ``ego_to_global`` を属性で、``Tracking3DInput`` はプロパティで公開し、
        ``ImageSample`` / ``Tracking2DInput`` は**持たない**（2D 系列に ego pose を載せない）。

        Returns:
            ``timestamps`` は ``(T,)`` の ``int64`` 配列。``ego_to_global`` は
            **全入力が公開する場合のみ** ``(T, 4, 4)`` 配列、そうでなければ ``None``。
            空の系列は正常とし、空の ``timestamps`` と ``None`` を返す。

        Raises:
            StreamingContractError: ``.timestamp`` を持たない入力、または
                ``timestamp`` が ``None`` の入力が混ざっている場合。**ループの前に**送出する。
        """
        if not inputs:
            return np.empty((0,), dtype=np.int64), None

        timestamps: list[int] = []
        for i, item in enumerate(inputs):
            if not hasattr(item, "timestamp"):
                raise StreamingContractError(
                    f"{type(self).__name__}: input at index {i} of type "
                    f"{type(item).__name__} does not expose '.timestamp'; streaming input "
                    "types must expose it (streaming_agents.md 3.2)."
                )
            ts = item.timestamp
            if ts is None:
                raise StreamingContractError(
                    f"{type(self).__name__}: input at index {i} has timestamp=None; every "
                    "frame in a streaming sequence needs a timestamp (evaluation uses it for "
                    "non-uniform frame intervals). Set it upstream before predict()."
                )
            timestamps.append(ts)

        egos = [getattr(item, "ego_to_global", None) for item in inputs]
        if all(e is not None for e in egos):
            ego_poses: np.ndarray | None = np.stack([np.asarray(e, dtype=np.float64) for e in egos])
        else:
            ego_poses = None
        return np.asarray(timestamps, dtype=np.int64), ego_poses

    def _check_requirements(self, inputs: list) -> None:
        """複合入力型の上流出力（``detections``）が要件を満たすか検証する（タスク4）。

        ``config.requires`` に上流タスクが宣言されているのに ``detections`` が ``None`` なら
        エラーにする。**実行時検証は agents の責務**であり、core は config と型を持つだけ
        （streaming_agents.md 5.2）。座標系も併せて検証する（3D の ``detections`` は ``EGO`` 前提。5.1）。

        ``predict`` の先頭で呼ぶことで、オフライン評価・server 一括の経路（既定 ``predict`` が
        制御する）では作者の規律に依存せず必ず走る。``step`` を直接呼ぶ経路（interfaces）でも
        守るために protected として公開しており、その経路の実装時に同じ検証を呼べる。

        Raises:
            UpstreamInputError: ``requires`` があるのに ``detections`` が ``None``、または
                ``detections`` の座標系が ``EGO`` でない場合。
        """
        config = getattr(self, "config", None)
        requires = getattr(config, "requires", None) or []
        if not requires:
            return
        for i, item in enumerate(inputs):
            if not hasattr(item, "detections"):
                # E2E / seg-tracking の入力型は上流の枠を持たない。
                continue
            detections = item.detections
            if detections is None:
                raise UpstreamInputError(
                    f"{type(self).__name__}: config.requires={list(requires)} but input at "
                    f"index {i} has detections=None; supply the upstream output (e.g. from a "
                    "detector run) before predict()."
                )
            frame = getattr(detections, "frame", None)
            if frame is not None and frame != CoordinateFrame.EGO:
                raise UpstreamInputError(
                    f"{type(self).__name__}: detections at index {i} are in frame "
                    f"{frame.value!r}, but EGO is required. Have the caller align them to EGO, "
                    "or call detections.to_global(sample.ego_to_global) inside the agent "
                    "(streaming_agents.md 5.1)."
                )

    @abstractmethod
    def _aggregate(
        self, frames: list, timestamps: np.ndarray, ego_to_global: np.ndarray | None
    ) -> Any:
        """:meth:`step` の出力列を系列の出力型にまとめる。

        **Agent 作者は実装しない。** ストリーミング用のタスク別抽象クラス
        （:class:`Tracking3DAgent` 等）が実装し、系列出力型へ詰め替える
        （streaming_agents.md 3.3）。

        系列の出力型は、単体タスクの出力型を時刻順に保持する薄いラッパとする
        （中間出力を単体タスクの出力型で表す既存方針と同じ）。これにより
        1 フレーム取り出せば既存の可視化コードがそのまま使える。

        ``timestamps`` は既定 :meth:`predict` が入力から引き写すため、**Agent 作者の
        負担にはならない**。``ego_to_global`` は 3D 系列でのみ意味を持ち、
        **2D 系列では ``None`` で渡る**（2D の ``_aggregate`` は無視する）。
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


# --- 単発のタスク別抽象クラス -------------------------------------------------
#
# predict のシグネチャで**出力型を固定**する。TaskType と出力型の 1 対 1 対応を
# 型の上で表現するのが目的であり、実装は持たない。これらはすべて**単発**
# （predict(input: Sample)）であり、StreamingMixin とは併用しない。
#
# ストリーミングが必要なタスク（tracking や時系列 E2E）は、StreamingMixin と
# BaseAgent の組（class X(StreamingMixin, BaseAgent)）による**専用のタスク別抽象クラス**を
# このファイル末尾に別途定義する（CLAUDE.md 2.3 / 4 章）。


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
    """End-to-end 自動運転（``sensing_to_planning``）。**単発推論**の抽象クラス。

    時系列モデル（UniAD 等）は本クラスを**継承しない**。``predict`` が系列を取るため
    シグネチャが非互換になるからである。ステートフルな E2E は :class:`StreamingE2EAgent`
    （本ファイル末尾。``StreamingMixin`` + ``BaseAgent``）を継承する。単発版と系列版は
    同一タスク（``sensing_to_planning``）の単発／系列という関係で、評価指標も同じである
    （streaming_agents.md 6.4）。
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


# --- ストリーミング用のタスク別抽象クラス -------------------------------------
#
# StreamingMixin + BaseAgent の組（単発クラスは継承しない）。系列出力型への詰め替え
# （_aggregate）を**抽象クラスが実装**するため、Agent 作者は reset / step だけを書けばよい
# （streaming_agents.md 3.3）。predict は StreamingMixin の既定実装をそのまま使い、
# **override しない**。戻り値の具体型は TYPE_CHECKING ブロックのシグネチャ宣言のみで
# mypy に伝える（実装は差し替えない）。


class Tracking3DAgent(StreamingMixin, BaseAgent):
    """3D 物体追跡（``object_tracking_3d``）。

    ``step`` は現フレームの ``Tracking3DInput``（``sample`` + 任意の上流 ``detections``）を
    取り、``Detection3DOutput``（``track_id`` 付き）を返す。既定 ``predict`` が系列を
    ``Detection3DSequence`` にまとめ、``timestamps`` / ``ego_to_global`` を入力から引き写す。
    """

    task = TaskType.OBJECT_TRACKING_3D

    if TYPE_CHECKING:
        # 実装は StreamingMixin.predict（override しない）。ここは戻り値型の宣言だけ。
        def predict(self, inputs: list[Tracking3DInput]) -> Detection3DSequence: ...

    def _aggregate(
        self, frames: list, timestamps: np.ndarray, ego_to_global: np.ndarray | None
    ) -> Detection3DSequence:
        return Detection3DSequence(
            frames=frames, timestamps=timestamps, ego_to_global=ego_to_global
        )

    @abstractmethod
    def step(self, input: Tracking3DInput) -> Detection3DOutput: ...


class Tracking2DAgent(StreamingMixin, BaseAgent):
    """2D 物体追跡（``object_tracking_2d``）。

    ``step`` は ``Tracking2DInput``（``image_sample`` + 任意の上流 ``detections``）を取り、
    ``Detection2DOutput`` を返す。2D 系列は ego pose を持たないため、``_aggregate`` は
    ``ego_to_global``（``None`` で渡る）を**無視する**。
    """

    task = TaskType.OBJECT_TRACKING_2D

    if TYPE_CHECKING:

        def predict(self, inputs: list[Tracking2DInput]) -> Detection2DSequence: ...

    def _aggregate(
        self, frames: list, timestamps: np.ndarray, ego_to_global: np.ndarray | None
    ) -> Detection2DSequence:
        # ego_to_global は 2D 系列では意味を持たないため無視する。
        return Detection2DSequence(frames=frames, timestamps=timestamps)

    @abstractmethod
    def step(self, input: Tracking2DInput) -> Detection2DOutput: ...


class InstanceSegmentationTracking2DAgent(StreamingMixin, BaseAgent):
    """インスタンスセグメンテーション追跡（``instance_segmentation_tracking_2d``）。

    入力は複合入力型を作らず ``ImageSample`` を直接取る。SAM2 系は上流検出を取らず、
    プロンプトは ``ImageSample.prompt`` で受けるためである（streaming_agents.md 5.3）。
    2D 系列のため ``_aggregate`` は ``ego_to_global``（``None`` で渡る）を**無視する**。
    """

    task = TaskType.INSTANCE_SEGMENTATION_TRACKING_2D

    if TYPE_CHECKING:

        def predict(self, inputs: list[ImageSample]) -> InstanceSegmentation2DSequence: ...

    def _aggregate(
        self, frames: list, timestamps: np.ndarray, ego_to_global: np.ndarray | None
    ) -> InstanceSegmentation2DSequence:
        # ego_to_global は 2D 系列では意味を持たないため無視する。
        return InstanceSegmentation2DSequence(frames=frames, timestamps=timestamps)

    @abstractmethod
    def step(self, input: ImageSample) -> InstanceSegmentation2DOutput: ...


class StreamingE2EAgent(StreamingMixin, BaseAgent):
    """ステートフルな End-to-end 自動運転（``sensing_to_planning`` の系列版）。

    UniAD / SparseDrive 等の時系列モデル。単発版 :class:`E2EAgent` を**継承しない**
    （``predict`` のシグネチャが単発と非互換なため。streaming_agents.md 4 章）。
    単発版と系列版は同一タスクの単発／系列という関係で、評価指標も同じである（6.4）。

    ``step`` は現フレームの ``Sample`` を取り ``E2EOutput`` を返す。既定 ``predict`` が
    系列を ``E2ESequence`` にまとめ、``timestamps`` / ``ego_to_global`` を引き写す。
    """

    task = TaskType.SENSING_TO_PLANNING

    if TYPE_CHECKING:

        def predict(self, inputs: list[Sample]) -> E2ESequence: ...

    def _aggregate(
        self, frames: list, timestamps: np.ndarray, ego_to_global: np.ndarray | None
    ) -> E2ESequence:
        return E2ESequence(frames=frames, timestamps=timestamps, ego_to_global=ego_to_global)

    @abstractmethod
    def step(self, input: Sample) -> E2EOutput: ...
