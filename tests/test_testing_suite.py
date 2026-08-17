"""共通テストスイート（``jidohub.agents.testing``）自体の検証。

準拠ダミーで契約チェックが通ること、違反ダミーを**期待通り検出**することを確認する
（依頼書 タスク 5.4）。
"""

from __future__ import annotations

import numpy as np
import pytest
from jidohub.core.schemas import Image, ImageSample, Sample, Tracking2DInput, Tracking3DInput

from jidohub.agents.testing import (
    check_2d_output_frame,
    check_agent_contract,
    check_mro_order,
    check_output_type,
    check_streaming_contract,
    check_timestamp_guard,
)
from tests.dummies import (
    DummyClassification2DAgent,
    DummyDetection2DAgent,
    DummyDetection3DAgent,
    DummyTracking2DAgent,
    DummyTracking3DAgent,
    LeakyStreamingAgent,
    ModelSpaceDetection2DAgent,
    NoGuardStreamingAgent,
    ReversedMroAgent,
    WrongOutputTypeAgent,
)

# --- 準拠ダミーは通る -------------------------------------------------------


def _make(agent_cls):  # type: ignore[no-untyped-def]
    return agent_cls._from_config(None, None, "cpu")


def test_stateless_agent_passes(sensor_sample: Sample) -> None:
    check_agent_contract(_make(DummyDetection3DAgent), sensor_sample)


def test_streaming_agent_passes(tracking3d_inputs: list[Tracking3DInput]) -> None:
    # Tracking3DAgent を継承したダミー（_aggregate を書かない）が契約を満たす。
    check_agent_contract(_make(DummyTracking3DAgent), tracking3d_inputs)


def test_streaming_3d_collects_ego(tracking3d_inputs: list[Tracking3DInput]) -> None:
    # 3D 系列は入力の ego_to_global を (T,4,4) に収集する。
    agent = _make(DummyTracking3DAgent)
    seq = agent.predict(tracking3d_inputs)
    assert seq.ego_to_global is not None
    assert seq.ego_to_global.shape == (len(tracking3d_inputs), 4, 4)
    assert np.array_equal(seq.timestamps, np.array([0, 1, 2], dtype=np.int64))


def test_streaming_2d_ego_is_none(tracking2d_inputs: list[Tracking2DInput]) -> None:
    # 2D 系列は ego pose を持たないため ego_to_global は None。
    agent = _make(DummyTracking2DAgent)
    check_agent_contract(agent, tracking2d_inputs)
    seq = agent.predict(tracking2d_inputs)
    # Detection2DSequence は ego_to_global フィールドを持たない（getattr で None）。
    assert getattr(seq, "ego_to_global", None) is None
    assert np.array_equal(seq.timestamps, np.array([10, 20], dtype=np.int64))


def test_timestamp_guard_stops_before_loop() -> None:
    # timestamp=None を混ぜると predict はループ前にエラーになり step は呼ばれない。
    agent = _make(DummyTracking2DAgent)
    inputs = [
        Tracking2DInput(
            image_sample=ImageSample(
                image=Image(pixels=np.zeros((10, 10, 3), dtype=np.uint8)), timestamp=1
            )
        ),
        Tracking2DInput(
            image_sample=ImageSample(
                image=Image(pixels=np.zeros((10, 10, 3), dtype=np.uint8)), timestamp=None
            )
        ),
    ]
    check_timestamp_guard(agent, inputs)


def test_2d_agent_passes(image_sample: ImageSample) -> None:
    check_agent_contract(_make(DummyDetection2DAgent), image_sample)


def test_classification_agent_passes(image_sample: ImageSample) -> None:
    check_output_type(_make(DummyClassification2DAgent), image_sample)


# --- 違反ダミーは検出される -------------------------------------------------


def test_detects_reversed_mro() -> None:
    with pytest.raises(AssertionError, match="StreamingMixin"):
        check_mro_order(ReversedMroAgent)


def test_detects_wrong_output_type(sensor_sample: Sample) -> None:
    with pytest.raises(AssertionError, match="requires Detection3DOutput"):
        check_output_type(_make(WrongOutputTypeAgent), sensor_sample)


def test_detects_missing_state_guard(tracking3d_inputs: list[Tracking3DInput]) -> None:
    with pytest.raises(AssertionError, match="StateNotInitializedError"):
        check_streaming_contract(_make(NoGuardStreamingAgent), tracking3d_inputs)


def test_detects_state_leak(tracking3d_inputs: list[Tracking3DInput]) -> None:
    with pytest.raises(AssertionError, match="manual reset"):
        check_streaming_contract(_make(LeakyStreamingAgent), tracking3d_inputs)


def test_detects_model_space_coordinates(image_sample: ImageSample) -> None:
    with pytest.raises(AssertionError, match="exceeds the input image"):
        check_2d_output_frame(_make(ModelSpaceDetection2DAgent), image_sample)
