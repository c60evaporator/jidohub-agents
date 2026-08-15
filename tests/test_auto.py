"""``AutoAgent``（native + inprocess のみ）の検証。

セキュリティ境界（隔離要件）を破らないこと、次段階の経路が明示的なエラーになることを確認する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
from jidohub.core.config import load_agent_config

from jidohub.agents import registry
from jidohub.agents.auto import AutoAgent
from jidohub.agents.exceptions import AgentResolutionError, IsolationViolationError
from tests.dummies import DummyDetection3DAgent


@pytest.fixture
def register_dummy(monkeypatch: pytest.MonkeyPatch) -> None:
    """``NATIVE_AGENTS`` にダミーを一時登録する（レジストリ本体は空のまま）。"""
    monkeypatch.setitem(registry.NATIVE_AGENTS, "DummyDetection3DAgent", DummyDetection3DAgent)


def _remote_code_config(schema_version: str) -> dict:
    return {
        "schema_version": schema_version,
        "agent_id": "test/Remote",
        "task": "object_detection_3d",
        "sensors": {"lidar": ["LIDAR_TOP"]},
        "implementation": {
            "type": "remote_code",
            "auto_map": {"AutoAgent": "src/modeling.py:RemoteAgent"},
        },
        "runtime": {"isolation": "required", "dockerfile": "Dockerfile"},
    }


def test_resolves_and_delegates(register_dummy: None, write_repo: Callable[..., Path]) -> None:
    repo = write_repo()
    agent = AutoAgent.from_pretrained(repo)
    assert isinstance(agent, DummyDetection3DAgent)


def test_reference_revision_preserved(
    register_dummy: None, write_repo: Callable[..., Path]
) -> None:
    # AutoAgent は取得のためローカル repo_path を委譲先へ渡すが、revision を含む
    # 解決済み参照を _resolved_reference で伝えるため、構築後の Agent が revision を保持する。
    repo = write_repo()
    agent = AutoAgent.from_pretrained(repo, revision="v1")
    assert agent.reference is not None
    assert agent.reference.revision == "v1"


def test_task_mismatch_detected(
    register_dummy: None, write_repo: Callable[..., Path], base_config: Callable[..., dict]
) -> None:
    cfg = base_config(task="map_construction")  # native_class は Detection3D のまま
    repo = write_repo(cfg)
    with pytest.raises(AgentResolutionError, match="task mismatch"):
        AutoAgent.from_pretrained(repo)


def test_remote_code_not_implemented(write_repo: Callable[..., Path]) -> None:
    from jidohub.core.schemas import SCHEMA_VERSION

    repo = write_repo(_remote_code_config(SCHEMA_VERSION))
    # 既定 runner=auto → isolation required → docker（未実装）。
    with pytest.raises(NotImplementedError):
        AutoAgent.from_pretrained(repo)


def test_resolve_class_remote_code_not_implemented(write_repo: Callable[..., Path]) -> None:
    from jidohub.core.schemas import SCHEMA_VERSION

    repo = write_repo(_remote_code_config(SCHEMA_VERSION))
    config = load_agent_config(repo)
    with pytest.raises(NotImplementedError, match="remote_code"):
        AutoAgent._resolve_class(config, repo)


def test_docker_runner_not_implemented(
    register_dummy: None, write_repo: Callable[..., Path]
) -> None:
    repo = write_repo()
    with pytest.raises(NotImplementedError, match="docker"):
        AutoAgent.from_pretrained(repo, runner="docker")


def test_shm_native_not_implemented(register_dummy: None, write_repo: Callable[..., Path]) -> None:
    repo = write_repo()
    with pytest.raises(NotImplementedError, match="shm"):
        AutoAgent.from_pretrained(repo, transport="shm")


def test_shm_remote_code_rejected(write_repo: Callable[..., Path]) -> None:
    from jidohub.core.schemas import SCHEMA_VERSION

    repo = write_repo(_remote_code_config(SCHEMA_VERSION))
    with pytest.raises(IsolationViolationError, match="shm"):
        AutoAgent.from_pretrained(repo, transport="shm")


def test_isolation_required_inprocess_rejected(
    register_dummy: None, write_repo: Callable[..., Path], base_config: Callable[..., dict]
) -> None:
    cfg = base_config(runtime={"isolation": "required", "dockerfile": "Dockerfile"})
    repo = write_repo(cfg)
    with pytest.raises(IsolationViolationError):
        AutoAgent.from_pretrained(repo, runner="inprocess")
