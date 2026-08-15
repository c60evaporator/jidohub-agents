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
)
from jidohub.core.tasks import TaskType

from jidohub.agents.base import (
    BaseAgent,
    Classification2DAgent,
    Detection2DAgent,
    Detection3DAgent,
    MapConstructionAgent,
    StreamingMixin,
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


class DummyStreamingDetection3DAgent(StreamingMixin, BaseAgent):
    """ステートフルな準拠 Agent（``track_id`` を採番する）。

    タスク別抽象クラス（``Detection3DAgent``）は継承せず ``StreamingMixin`` +
    ``BaseAgent`` の組にする（``predict`` が系列を取り、単発の宣言と非互換のため）。
    """

    task = TaskType.OBJECT_DETECTION_3D

    @classmethod
    def _from_config(
        cls, config: AgentConfig, repo_path: Path, device: str, **kwargs: Any
    ) -> "DummyStreamingDetection3DAgent":
        return cls()

    def load_weights(self, path: Path) -> None:  # pragma: no cover - weights なし想定
        pass

    def reset(self) -> None:
        super().reset()
        self._counter = 0

    def step(self, input: Sample) -> Detection3DOutput:
        self._check_initialized()
        self._counter += 1
        return Detection3DOutput(boxes=[_box3d(track_id=self._counter)])

    def _aggregate(self, frames: list) -> list:
        # 系列出力型は単体出力の時刻順リスト（薄いラッパ）。
        return frames


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
    インスタンス化すると ``predict`` 未実装の ``TypeError`` になるため、
    型としてのみ使う。
    """

    def _aggregate(self, frames: list) -> list:  # pragma: no cover
        return frames


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


class NoGuardStreamingAgent(StreamingMixin, BaseAgent):
    """``step`` が ``_check_initialized`` を呼ばない違反 Agent。"""

    task = TaskType.OBJECT_DETECTION_3D

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

    def step(self, input: Sample) -> Detection3DOutput:
        # _check_initialized を呼ばない（前シーンの状態漏れを許す）。
        self._counter = getattr(self, "_counter", 0) + 1
        return Detection3DOutput(boxes=[_box3d(track_id=self._counter)])

    def _aggregate(self, frames: list) -> list:
        return frames


class LeakyStreamingAgent(StreamingMixin, BaseAgent):
    """``reset`` が内部カウンタをリセットしない違反 Agent。

    ``predict()`` と ``reset()+step()`` ループで ``track_id`` がずれる。
    """

    task = TaskType.OBJECT_DETECTION_3D
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

    def step(self, input: Sample) -> Detection3DOutput:
        self._check_initialized()
        self._counter += 1
        return Detection3DOutput(boxes=[_box3d(track_id=self._counter)])

    def _aggregate(self, frames: list) -> list:
        return frames


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
