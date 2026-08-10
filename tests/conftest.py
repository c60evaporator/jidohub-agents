"""共通 fixture。

``from_pretrained`` はローカル repo_path（``agent_config.json`` を含むディレクトリ）を
そのまま受け取れる（``HubClient.snapshot`` がローカル参照をコピーせず返すため）。
テストは一時ディレクトリに最小の config を書き出して駆動する。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
from jidohub.core.schemas import SCHEMA_VERSION, Image, ImageSample, LidarSweep, Sample


def _base_config(**overrides: Any) -> dict:
    """object_detection_3d の native Agent の最小 config。"""
    config = {
        "schema_version": SCHEMA_VERSION,
        "agent_id": "test/Dummy",
        "task": "object_detection_3d",
        "sensors": {"lidar": ["LIDAR_TOP"]},
        "implementation": {"type": "native", "native_class": "DummyDetection3DAgent"},
        "runtime": {"isolation": "not-required", "gpu_required": False},
        "weights": [],
    }
    config.update(overrides)
    return config


@pytest.fixture
def write_repo(tmp_path: Path) -> Callable[..., Path]:
    """config（+ 任意で重みファイル）を書いた repo ディレクトリを作るファクトリ。"""

    def _make(config: dict | None = None, weight_bytes: bytes | None = None) -> Path:
        repo = tmp_path / f"repo_{len(list(tmp_path.iterdir()))}"
        repo.mkdir()
        cfg = config if config is not None else _base_config()
        if weight_bytes is not None:
            (repo / "weights").mkdir()
            (repo / "weights" / "model.safetensors").write_bytes(weight_bytes)
            sha = hashlib.sha256(weight_bytes).hexdigest()
            cfg = dict(cfg)
            cfg["weights"] = [
                {
                    "path": "weights/model.safetensors",
                    "format": "safetensors",
                    "sha256": sha,
                }
            ]
        (repo / "agent_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        return repo

    return _make


@pytest.fixture
def base_config() -> Callable[..., dict]:
    return _base_config


@pytest.fixture
def sensor_sample() -> Sample:
    """LiDAR を 1 つ持つ最小の Sample。"""
    return Sample(
        timestamp=0,
        ego_to_global=np.eye(4),
        lidar=LidarSweep(
            points=np.zeros((1, 4), dtype=np.float32),
            sensor_to_ego=np.eye(4),
        ),
    )


@pytest.fixture
def image_sample() -> ImageSample:
    """200x100 の画素を持つ ImageSample。"""
    return ImageSample(image=Image(pixels=np.zeros((100, 200, 3), dtype=np.uint8)))
