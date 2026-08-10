"""共通テストスイート（``jidohub.agents.testing``）自体の検証。

準拠ダミーで契約チェックが通ること、違反ダミーを**期待通り検出**することを確認する
（依頼書 タスク 5.4）。
"""

from __future__ import annotations

import pytest
from jidohub.core.schemas import ImageSample, Sample

from jidohub.agents.testing import (
    check_2d_output_frame,
    check_agent_contract,
    check_mro_order,
    check_output_type,
    check_streaming_contract,
)
from tests.dummies import (
    DummyClassification2DAgent,
    DummyDetection2DAgent,
    DummyDetection3DAgent,
    DummyStreamingDetection3DAgent,
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


def test_streaming_agent_passes(sensor_sample: Sample) -> None:
    agent = _make(DummyStreamingDetection3DAgent)
    check_agent_contract(agent, [sensor_sample, sensor_sample, sensor_sample])


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


def test_detects_missing_state_guard(sensor_sample: Sample) -> None:
    with pytest.raises(AssertionError, match="StateNotInitializedError"):
        check_streaming_contract(_make(NoGuardStreamingAgent), [sensor_sample, sensor_sample])


def test_detects_state_leak(sensor_sample: Sample) -> None:
    with pytest.raises(AssertionError, match="manual reset"):
        check_streaming_contract(_make(LeakyStreamingAgent), [sensor_sample, sensor_sample])


def test_detects_model_space_coordinates(image_sample: ImageSample) -> None:
    with pytest.raises(AssertionError, match="exceeds the input image"):
        check_2d_output_frame(_make(ModelSpaceDetection2DAgent), image_sample)
