"""契約を満たす／破るダミー Agent。

共通テストスイート（:mod:`jidohub.agents.testing`）自体を検証するために使う。
**準拠ダミー**（ステートレス / ステートフル）と、共通スイートが検出すべき
**違反ダミー**の両方を用意する（依頼書 タスク 5.4）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from jidohub.core.config import AgentConfig
from jidohub.core.schemas import (
    Box2D,
    Box3D,
    Classification2DOutput,
    CoordinateFrame,
    Detection2DOutput,
    Detection3DOutput,
    ImageSample,
    MapOutput,
    Sample,
    Tracking2DInput,
    Tracking3DInput,
)

from jidohub.agents.base import (
    BaseAgent,
    Classification2DAgent,
    Detection2DAgent,
    Detection3DAgent,
    MapConstructionAgent,
    StreamingMixin,
    Tracking2DAgent,
    Tracking3DAgent,
)


def _box3d(track_id: int | None = None) -> Box3D:
    return Box3D(
        center=np.zeros(3),
        size=np.ones(3),
        rotation=np.array([1.0, 0.0, 0.0, 0.0]),
        label="car",
        track_id=track_id,
    )


# --- 準拠ダミー -------------------------------------------------------------


class DummyDetection3DAgent(Detection3DAgent):
    """ステートレスな準拠 Agent。"""

    #: 動的属性は from_config で設定する。クラス注釈で mypy に知らせる。
    loaded_weights: list[Path]
    from_config_kwargs: dict[str, Any]

    @classmethod
    def _from_config(
        cls, config: AgentConfig, repo_path: Path, device: str, **kwargs: Any
    ) -> "DummyDetection3DAgent":
        obj = cls()
        obj.loaded_weights = []
        obj.from_config_kwargs = kwargs
        return obj

    def load_weights(self, path: Path) -> None:
        self.loaded_weights.append(path)

    def predict(self, input: Sample) -> Detection3DOutput:
        return Detection3DOutput(boxes=[_box3d()], frame=CoordinateFrame.EGO)


class DummyTracking3DAgent(Tracking3DAgent):
    """ステートフルな準拠 Agent（``track_id`` を採番する）。

    ストリーミング用のタスク別抽象クラス（:class:`Tracking3DAgent`）を継承するため、
    **系列の集約を書かず** ``reset`` / ``step`` のみ実装する（これが今回の主目的の実証）。
    """

    @classmethod
    def _from_config(
        cls, config: AgentConfig, repo_path: Path, device: str, **kwargs: Any
    ) -> "DummyTracking3DAgent":
        return cls()

    def load_weights(self, path: Path) -> None:  # pragma: no cover - weights なし想定
        pass

    def reset(self) -> None:
        super().reset()
        self._counter = 0

    def step(self, input: Tracking3DInput) -> Detection3DOutput:
        self._check_initialized()
        self._counter += 1
        return Detection3DOutput(boxes=[_box3d(track_id=self._counter)])


class DummyTracking2DAgent(Tracking2DAgent):
    """2D 追跡の準拠 Agent。系列の集約は抽象クラスが引き受ける。

    2D 系列は ego pose を持たないため、系列出力の ``ego_to_global`` が ``None`` になることの
    検証に使う。``step`` の呼び出し回数（``step_calls``）を数え、
    :func:`~jidohub.agents.testing.check_timestamp_guard` の検証にも用いる。
    """

    #: step の呼び出し回数。タイムスタンプ欠落時に step がループ前で止まることの確認に使う。
    step_calls: int

    @classmethod
    def _from_config(
        cls, config: AgentConfig, repo_path: Path, device: str, **kwargs: Any
    ) -> "DummyTracking2DAgent":
        obj = cls()
        obj.step_calls = 0
        return obj

    def load_weights(self, path: Path) -> None:  # pragma: no cover - weights なし想定
        pass

    def reset(self) -> None:
        super().reset()
        self._counter = 0

    def step(self, input: Tracking2DInput) -> Detection2DOutput:
        self._check_initialized()
        self.step_calls += 1
        self._counter += 1
        box = Box2D(xyxy=np.array([0.0, 0.0, 1.0, 1.0]), label="car", score=0.9)
        return Detection2DOutput(boxes=[box], normalized=False)


class DummyDetection2DAgent(Detection2DAgent):
    """入力画像サイズ基準の座標を返す準拠 2D Agent。"""

    @classmethod
    def _from_config(
        cls, config: AgentConfig, repo_path: Path, device: str, **kwargs: Any
    ) -> "DummyDetection2DAgent":
        return cls()

    def load_weights(self, path: Path) -> None:  # pragma: no cover
        pass

    def predict(self, input: ImageSample) -> Detection2DOutput:
        w, h = float(input.image.width), float(input.image.height)
        box = Box2D(xyxy=np.array([0.0, 0.0, w / 2.0, h / 2.0]), label="car", score=0.9)
        return Detection2DOutput(boxes=[box], normalized=False)


# --- 違反ダミー（共通スイートが検出すべき） ---------------------------------


class ReversedMroAgent(BaseAgent, StreamingMixin):
    """継承順序が逆（``BaseAgent`` が ``StreamingMixin`` より前）。

    :func:`~jidohub.agents.testing.check_mro_order` が検出する。
    抽象のままインスタンス化せず、**型としてのみ**使う（ストリーミング用の
    タスク別抽象クラスは継承順序を固定するため、逆順の再現には使えない）。
    """


class WrongOutputTypeAgent(Detection3DAgent):
    """タスク宣言（3D 検出）と異なる出力型を返す違反 Agent。"""

    @classmethod
    def _from_config(
        cls, config: AgentConfig, repo_path: Path, device: str, **kwargs: Any
    ) -> "WrongOutputTypeAgent":
        return cls()

    def load_weights(self, path: Path) -> None:  # pragma: no cover
        pass

    def predict(self, input: Sample) -> Any:
        return MapOutput()  # Detection3DOutput であるべき。


class NoGuardStreamingAgent(Tracking3DAgent):
    """``step`` が ``_check_initialized`` を呼ばない違反 Agent。

    :class:`Tracking3DAgent` を継承するため系列の集約は書かない。
    """

    @classmethod
    def _from_config(
        cls, config: AgentConfig, repo_path: Path, device: str, **kwargs: Any
    ) -> "NoGuardStreamingAgent":
        return cls()

    def load_weights(self, path: Path) -> None:  # pragma: no cover
        pass

    def reset(self) -> None:
        super().reset()
        self._counter = 0

    def step(self, input: Tracking3DInput) -> Detection3DOutput:
        # _check_initialized を呼ばない（前シーンの状態漏れを許す）。
        self._counter = getattr(self, "_counter", 0) + 1
        return Detection3DOutput(boxes=[_box3d(track_id=self._counter)])


class LeakyStreamingAgent(Tracking3DAgent):
    """``reset`` が内部カウンタをリセットしない違反 Agent。

    ``predict()`` と ``reset()+step()`` ループで ``track_id`` がずれる。
    :class:`Tracking3DAgent` を継承するため系列の集約は書かない。
    """

    _counter = 0

    @classmethod
    def _from_config(
        cls, config: AgentConfig, repo_path: Path, device: str, **kwargs: Any
    ) -> "LeakyStreamingAgent":
        return cls()

    def load_weights(self, path: Path) -> None:  # pragma: no cover
        pass

    def reset(self) -> None:
        super().reset()
        # カウンタをリセットしない（意図的な違反）。

    def step(self, input: Tracking3DInput) -> Detection3DOutput:
        self._check_initialized()
        self._counter += 1
        return Detection3DOutput(boxes=[_box3d(track_id=self._counter)])


class ModelSpaceDetection2DAgent(Detection2DAgent):
    """出力座標を入力画像基準へ戻し忘れた違反 2D Agent（座標が範囲外）。"""

    @classmethod
    def _from_config(
        cls, config: AgentConfig, repo_path: Path, device: str, **kwargs: Any
    ) -> "ModelSpaceDetection2DAgent":
        return cls()

    def load_weights(self, path: Path) -> None:  # pragma: no cover
        pass

    def predict(self, input: ImageSample) -> Detection2DOutput:
        # 入力画像より遥かに大きいモデル入力空間の座標のまま返す。
        box = Box2D(xyxy=np.array([0.0, 0.0, 4096.0, 4096.0]), label="car")
        return Detection2DOutput(boxes=[box], normalized=False)


class DummyClassification2DAgent(Classification2DAgent):
    """プロンプト（ラベル候補）を使う準拠 Agent。"""

    @classmethod
    def _from_config(
        cls, config: AgentConfig, repo_path: Path, device: str, **kwargs: Any
    ) -> "DummyClassification2DAgent":
        return cls()

    def load_weights(self, path: Path) -> None:  # pragma: no cover
        pass

    def predict(self, input: ImageSample) -> Classification2DOutput:
        return Classification2DOutput(labels=["car"], scores=np.array([1.0]))


class DummyMapAgent(MapConstructionAgent):
    """重み・ルールベース想定の準拠 Agent。"""

    @classmethod
    def _from_config(
        cls, config: AgentConfig, repo_path: Path, device: str, **kwargs: Any
    ) -> "DummyMapAgent":
        return cls()

    def load_weights(self, path: Path) -> None:  # pragma: no cover
        pass

    def predict(self, input: Sample) -> MapOutput:
        return MapOutput()
