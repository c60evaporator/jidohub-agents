"""``StreamingMixin._check_requirements``（複合入力型の上流出力検証）の検証。

``config.requires`` が上流タスクを宣言しているのに ``detections`` が ``None``、または
座標系が ``EGO`` でない場合に ``UpstreamInputError`` になること（streaming_agents.md 5.1 / 5.2）。
検証は ``predict`` の先頭で行われる（オフライン評価経路を作者の規律に依存せず守るため）。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from jidohub.core.schemas import (
    CoordinateFrame,
    Detection3DOutput,
    Sample,
    Tracking3DInput,
)

from jidohub.agents.exceptions import UpstreamInputError
from tests.dummies import DummyTracking3DAgent


def _agent_requiring_detections() -> DummyTracking3DAgent:
    agent = DummyTracking3DAgent._from_config(None, None, "cpu")  # type: ignore[arg-type]
    # AgentConfig 全体を組まず、_check_requirements が読む .requires だけを持たせる。
    agent.config = SimpleNamespace(requires=["object_detection_3d"])  # type: ignore[assignment]
    return agent


def _input(detections: Detection3DOutput | None) -> Tracking3DInput:
    return Tracking3DInput(
        sample=Sample(timestamp=0, ego_to_global=np.eye(4)), detections=detections
    )


def test_missing_detections_rejected() -> None:
    agent = _agent_requiring_detections()
    with pytest.raises(UpstreamInputError, match="detections=None"):
        agent.predict([_input(None)])


def test_wrong_frame_rejected() -> None:
    agent = _agent_requiring_detections()
    detections = Detection3DOutput(frame=CoordinateFrame.GLOBAL)
    with pytest.raises(UpstreamInputError, match="EGO"):
        agent.predict([_input(detections)])


def test_ego_detections_accepted() -> None:
    agent = _agent_requiring_detections()
    detections = Detection3DOutput(frame=CoordinateFrame.EGO)
    seq = agent.predict([_input(detections)])
    assert len(seq.frames) == 1


def test_no_requires_skips_check() -> None:
    # config.requires が空なら detections=None でも検証しない。
    agent = DummyTracking3DAgent._from_config(None, None, "cpu")  # type: ignore[arg-type]
    seq = agent.predict([_input(None)])
    assert len(seq.frames) == 1
