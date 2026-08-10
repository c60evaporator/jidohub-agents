"""``BaseAgent.from_pretrained``（テンプレートメソッド）の検証。"""

from __future__ import annotations

from typing import Callable

import pytest
from jidohub.core.schemas import Detection3DOutput, Sample

from jidohub.agents.exceptions import AgentResolutionError, IsolationViolationError
from tests.dummies import DummyDetection3DAgent


def test_loads_native_agent(write_repo: Callable[..., object], sensor_sample: Sample) -> None:
    repo = write_repo()
    agent = DummyDetection3DAgent.from_pretrained(repo, device="cpu")
    assert isinstance(agent, DummyDetection3DAgent)
    assert agent.repo_path == repo
    assert agent.device == "cpu"
    assert agent.config.agent_id == "test/Dummy"
    assert isinstance(agent.predict(sensor_sample), Detection3DOutput)


def test_task_mismatch_detected(
    write_repo: Callable[..., object], base_config: Callable[..., dict]
) -> None:
    # config は map_construction を宣言するが、クラスは object_detection_3d を実装。
    cfg = base_config(task="map_construction")
    repo = write_repo(cfg)
    with pytest.raises(AgentResolutionError, match="task mismatch"):
        DummyDetection3DAgent.from_pretrained(repo)


def test_isolation_required_rejected(
    write_repo: Callable[..., object], base_config: Callable[..., dict]
) -> None:
    cfg = base_config(runtime={"isolation": "required", "dockerfile": "Dockerfile"})
    repo = write_repo(cfg)
    with pytest.raises(IsolationViolationError, match="isolation='required'"):
        DummyDetection3DAgent.from_pretrained(repo)


def test_weights_loaded(write_repo: Callable[..., object]) -> None:
    repo = write_repo(weight_bytes=b"dummy-weights")
    agent = DummyDetection3DAgent.from_pretrained(repo)
    assert agent.loaded_weights == [repo / "weights" / "model.safetensors"]


def test_no_weights_skips_load(write_repo: Callable[..., object]) -> None:
    repo = write_repo()  # weights: []
    agent = DummyDetection3DAgent.from_pretrained(repo)
    assert agent.loaded_weights == []


def test_from_config_receives_kwargs(write_repo: Callable[..., object]) -> None:
    repo = write_repo()
    agent = DummyDetection3DAgent.from_pretrained(repo, threshold=0.5)
    assert agent.from_config_kwargs == {"threshold": 0.5}
