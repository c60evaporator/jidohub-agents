"""Agent の共通テストスイート。

``BaseAgent`` を実装する Agent が満たすべき性質を**再利用可能な形**で提供する。
ネイティブ実装にも、外部の Agent 作者にも同じ検証を適用できるようにするため
（CLAUDE.md 3.1）。

Example:
    >>> from jidohub.agents.testing import check_agent_contract
    >>> check_agent_contract(agent, sample_input)

出力の一致比較（:func:`_outputs_equal`）
    ストリーミングの系列出力型（``_aggregate`` の戻り値。``Detection3DSequence`` 等）は
    ``pack`` に頼らず **再帰的な構造比較**で判定する。dataclass はフィールドごと、
    numpy 配列（``timestamps`` / ``ego_to_global``）は ``np.array_equal`` で比較するため、
    ``reset()`` + ``step()`` ループと ``predict()`` の**厳密一致**を検証できる。
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from jidohub.core.schemas import (
    Classification2DOutput,
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
)
from jidohub.core.tasks import TaskType

from jidohub.agents.base import BaseAgent, StreamingMixin
from jidohub.agents.exceptions import StateNotInitializedError, StreamingContractError

__all__ = [
    "check_agent_contract",
    "check_streaming_contract",
    "check_timestamp_guard",
    "check_mro_order",
    "check_output_type",
    "check_2d_output_frame",
]

# TaskType → 単体推論（1 フレーム）の出力型。base.py のタスク別抽象クラスと 1 対 1。
# 予約タスク（出力型未定義）は含めない。
_TASK_OUTPUT_TYPES: dict[TaskType, type] = {
    TaskType.OBJECT_DETECTION_3D: Detection3DOutput,
    TaskType.MAP_CONSTRUCTION: MapOutput,
    TaskType.SENSING_TO_PLANNING: E2EOutput,
    TaskType.OBJECT_DETECTION_2D: Detection2DOutput,
    TaskType.INSTANCE_SEGMENTATION_2D: InstanceSegmentation2DOutput,
    TaskType.IMAGE_CLASSIFICATION_2D: Classification2DOutput,
}

# TaskType → 系列推論（ストリーミング）の出力型。base.py のストリーミング用抽象クラスと 1 対 1。
_STREAMING_SEQUENCE_TYPES: dict[TaskType, type] = {
    TaskType.OBJECT_TRACKING_3D: Detection3DSequence,
    TaskType.OBJECT_TRACKING_2D: Detection2DSequence,
    TaskType.INSTANCE_SEGMENTATION_TRACKING_2D: InstanceSegmentation2DSequence,
    TaskType.SENSING_TO_PLANNING: E2ESequence,
}

# 2D 出力座標の範囲検証で許容する画素の余裕（丸め・境界の 1 画素ずれを吸収する）。
_PIXEL_TOLERANCE = 1.0


def _outputs_equal(a: Any, b: Any) -> bool:
    """2 つの出力を再帰的に構造比較する。

    dataclass はフィールドごと、list/tuple/dict は要素ごと、``np.ndarray`` は
    :func:`numpy.array_equal`、enum やスカラは ``==`` で比較する。
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        return a.shape == b.shape and bool(np.array_equal(a, b))
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        if type(a) is not type(b):
            return False
        return all(
            _outputs_equal(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a)
        )
    if isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return False
        return all(_outputs_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            return False
        return all(_outputs_equal(a[k], b[k]) for k in a)
    return bool(a == b)


def check_mro_order(agent_class: type) -> None:
    """``StreamingMixin`` が ``BaseAgent`` より前に来ることを検証する。

    逆順だと Mixin の ``predict`` が abstract を実装したことにならず、
    **インスタンス化時に「predict が未実装」という TypeError** になる。
    落ちること自体は正しいが、``reset`` / ``step`` / ``_aggregate`` を実装済みの
    作者にはエラー文から原因が読み取れないため、明示的なメッセージを出す。

    ``StreamingMixin`` を継承しないクラスでは何も検証しない（対象外）。
    """
    mro = agent_class.__mro__
    if StreamingMixin not in mro or BaseAgent not in mro:
        return
    if mro.index(StreamingMixin) > mro.index(BaseAgent):
        raise AssertionError(
            f"{agent_class.__name__} inherits (BaseAgent, StreamingMixin) but the order "
            "is reversed: BaseAgent precedes StreamingMixin in the MRO, so "
            "StreamingMixin.predict does not override BaseAgent's abstract predict and "
            "the class fails to instantiate with a misleading 'predict is abstract' "
            f"TypeError. Fix by declaring 'class {agent_class.__name__}(StreamingMixin, "
            "BaseAgent)' (mixin first)."
        )


def check_output_type(agent: Any, sample_input: Any) -> None:
    """``predict`` の出力型が ``TaskType`` の宣言と一致することを検証する。

    ストリーミング Agent の ``predict`` は系列出力を返すため対象外
    （:func:`check_streaming_contract` が扱う）。
    """
    task = agent.task
    if task not in _TASK_OUTPUT_TYPES:
        raise AssertionError(
            f"no output type is registered for task {task.value!r}; "
            "check_output_type supports only tasks with a defined output schema"
        )
    expected = _TASK_OUTPUT_TYPES[task]
    output = agent.predict(sample_input)
    if not isinstance(output, expected):
        raise AssertionError(
            f"{type(agent).__name__}.predict returned {type(output).__name__}, "
            f"but task {task.value!r} requires {expected.__name__}"
        )


def check_streaming_contract(agent: Any, sample_inputs: list) -> None:
    """ストリーミング Agent の不変条件を検証する（CLAUDE.md 2.3 / 3.1）。

    - ``predict(inputs)`` の戻り値が**系列出力型**（``frames`` / ``timestamps`` を持つ）である
    - ``timestamps`` が入力の ``.timestamp`` と一致する
    - ``ego_to_global`` が入力から正しく収集されている（3D）／``None`` である（2D）
    - ``predict(inputs)`` の結果が ``reset()`` + ``step()`` ループと**一致**する
    - 未 ``reset()`` での ``step()`` が :class:`StateNotInitializedError` になる
    - ``reset()`` が冪等である
    - ``predict()`` を 2 回実行しても結果が同じ（内部状態がリセットされている）

    一致しない場合、既定 ``predict()`` を上書きしているか、``step()`` が
    入力以外の状態に依存している。

    ``timestamp`` が ``None`` の入力でループ前にエラーになることは、汎用の入力に
    ``None`` を注入できないため :func:`check_timestamp_guard` で別途検証する。
    """
    if not isinstance(agent, StreamingMixin):
        raise AssertionError(
            f"{type(agent).__name__} is not a StreamingMixin agent; "
            "check_streaming_contract applies only to streaming agents"
        )

    # 未 reset での step() が明示的にエラーになること（前シーンの状態漏れの防止）。
    agent._initialized = False
    try:
        agent.step(sample_inputs[0])
    except StateNotInitializedError:
        pass
    else:
        raise AssertionError(
            f"{type(agent).__name__}.step() must raise StateNotInitializedError when "
            "called before reset(); step() implementations must call _check_initialized()"
        )

    predicted = agent.predict(sample_inputs)

    # 戻り値が系列出力型であること（薄いラッパ: frames + timestamps）。
    # task は BaseAgent の属性だが isinstance で StreamingMixin に絞られるため getattr で読む。
    task = getattr(agent, "task", None)
    expected_seq = _STREAMING_SEQUENCE_TYPES.get(task) if task is not None else None
    if expected_seq is not None and not isinstance(predicted, expected_seq):
        task_name = task.value if task is not None else "<unknown>"
        raise AssertionError(
            f"{type(agent).__name__}.predict returned {type(predicted).__name__}, but task "
            f"{task_name!r} requires the sequence type {expected_seq.__name__}"
        )
    if not (hasattr(predicted, "frames") and hasattr(predicted, "timestamps")):
        raise AssertionError(
            f"{type(agent).__name__}.predict must return a sequence output type with "
            f"'frames' and 'timestamps'; got {type(predicted).__name__}"
        )

    # timestamps が入力の .timestamp と一致すること。
    expected_ts = np.asarray([item.timestamp for item in sample_inputs], dtype=np.int64)
    if not (
        isinstance(predicted.timestamps, np.ndarray)
        and predicted.timestamps.shape == expected_ts.shape
        and np.array_equal(predicted.timestamps, expected_ts)
    ):
        raise AssertionError(
            f"{type(agent).__name__}: predict().timestamps "
            f"{getattr(predicted.timestamps, 'tolist', lambda: predicted.timestamps)()} do not "
            f"match the inputs' .timestamp {expected_ts.tolist()}"
        )

    # ego_to_global が入力から正しく収集されていること（全入力が持つ場合のみ (T,4,4)）。
    input_egos = [getattr(item, "ego_to_global", None) for item in sample_inputs]
    seq_ego = getattr(predicted, "ego_to_global", None)
    if all(e is not None for e in input_egos) and input_egos:
        expected_ego = np.stack([np.asarray(e, dtype=np.float64) for e in input_egos])
        if seq_ego is None or not np.array_equal(np.asarray(seq_ego), expected_ego):
            raise AssertionError(
                f"{type(agent).__name__}: predict().ego_to_global was not collected from the "
                f"inputs (expected shape {expected_ego.shape})"
            )
    elif seq_ego is not None:
        raise AssertionError(
            f"{type(agent).__name__}: predict().ego_to_global should be None when inputs do "
            f"not all expose ego_to_global (2D sequences), got shape {np.asarray(seq_ego).shape}"
        )

    # predict(inputs) == reset() + step() ループ。
    agent.reset()
    frames = [agent.step(item) for item in sample_inputs]
    timestamps, ego_poses = agent._collect_meta(sample_inputs)
    manual = agent._aggregate(frames, timestamps, ego_poses)
    if not _outputs_equal(predicted, manual):
        raise AssertionError(
            f"{type(agent).__name__}.predict(inputs) does not match a manual "
            "reset()+step() loop. The default predict() may be overridden, or step() "
            "depends on state other than its input."
        )

    # reset() が冪等（2 回呼んでも壊れず、以後 step() が動く）。
    agent.reset()
    agent.reset()
    agent.step(sample_inputs[0])

    # predict() を 2 回実行して結果が同じ（内部状態がリセットされている）。
    first = agent.predict(sample_inputs)
    second = agent.predict(sample_inputs)
    if not _outputs_equal(first, second):
        raise AssertionError(
            f"{type(agent).__name__}.predict() is not repeatable: two runs differ, "
            "so predict() leaks state across calls (reset() is not clearing everything)"
        )


def check_timestamp_guard(agent: Any, inputs_with_missing_timestamp: list) -> None:
    """``timestamp`` が欠けた入力で ``predict`` が**ループ前に**エラーになることを検証する。

    100 フレーム推論した後に集約で落ちるのを防ぐという契約（streaming_agents.md 3.2）を
    機械的に確認する。``step`` が **1 度も呼ばれない**ことを、呼び出し回数を数えるダミー
    （``step_calls`` 属性）で検証する。

    Args:
        agent: ``step_calls`` を 0 に持ち、``step`` の度に増やすストリーミングダミー。
        inputs_with_missing_timestamp: ``.timestamp`` が ``None`` の入力を 1 つ以上含む系列。
    """
    if not isinstance(agent, StreamingMixin):
        raise AssertionError(
            f"{type(agent).__name__} is not a StreamingMixin agent; "
            "check_timestamp_guard applies only to streaming agents"
        )
    # step_calls はダミー固有の計数属性（契約型には無い）ため getattr/setattr で扱う。
    setattr(agent, "step_calls", 0)
    try:
        agent.predict(inputs_with_missing_timestamp)
    except StreamingContractError:
        pass
    else:
        raise AssertionError(
            f"{type(agent).__name__}.predict must raise StreamingContractError when an input "
            "is missing its timestamp; the check must happen before the step() loop"
        )
    step_calls = getattr(agent, "step_calls", 0)
    if step_calls != 0:
        raise AssertionError(
            f"{type(agent).__name__}.predict called step() {step_calls} time(s) before "
            "detecting the missing timestamp; the guard must run before the loop so a long "
            "sequence does not run to completion and then fail on aggregation"
        )


def check_2d_output_frame(agent: Any, image_sample: Any) -> None:
    """2D Agent の出力座標が**入力 ``Image`` の現サイズ基準**であることを検証する。

    モデル入力用に resize したまま戻し忘れると、座標が数倍ずれる。例外は出ないため、
    入力サイズとの整合を機械的に確認する（CLAUDE.md 2.6）。

    非正規化なら box 座標が入力画像の範囲（0..width / 0..height）に収まること、
    正規化なら ``[0, 1]`` に収まることを検証する。
    """
    output = agent.predict(image_sample)
    image = image_sample.image
    width, height = float(image.width), float(image.height)

    boxes = _collect_boxes(output)
    normalized = getattr(output, "normalized", False)
    for box in boxes:
        xyxy = np.asarray(box.xyxy, dtype=np.float64)
        if normalized:
            if not (xyxy >= -1e-6).all() or not (xyxy <= 1.0 + 1e-6).all():
                raise AssertionError(
                    f"{type(agent).__name__}: normalized box {xyxy.tolist()} escapes "
                    "[0, 1]; coordinates were not mapped back to the input image"
                )
            continue
        x0, y0, x1, y1 = xyxy
        within = (
            xyxy.min() >= -_PIXEL_TOLERANCE
            and x0 <= width + _PIXEL_TOLERANCE
            and x1 <= width + _PIXEL_TOLERANCE
            and y0 <= height + _PIXEL_TOLERANCE
            and y1 <= height + _PIXEL_TOLERANCE
        )
        if not within:
            raise AssertionError(
                f"{type(agent).__name__}: box {xyxy.tolist()} exceeds the input image "
                f"size {int(width)}x{int(height)}; the agent likely left coordinates in "
                "model-input space instead of mapping them back (CLAUDE.md 2.6)"
            )


def _collect_boxes(output: Any) -> list:
    """2D 出力から :class:`~jidohub.core.schemas.Box2D` のリストを取り出す。"""
    if hasattr(output, "boxes"):
        return list(output.boxes)
    if hasattr(output, "instances"):
        return [inst.box for inst in output.instances]
    return []


def check_agent_contract(agent: Any, sample_input: Any) -> None:
    """Agent が契約を満たすことを、Agent の性質に応じて一括で検証する。"""
    check_mro_order(type(agent))
    if isinstance(agent, StreamingMixin):
        # 系列入力を前提とする。単一サンプルが渡された場合は 1 要素の系列に包む。
        inputs = sample_input if isinstance(sample_input, list) else [sample_input]
        check_streaming_contract(agent, inputs)
    else:
        check_output_type(agent, sample_input)
    if isinstance(sample_input, ImageSample):
        check_2d_output_frame(agent, sample_input)
