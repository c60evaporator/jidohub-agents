"""PEP 420 namespace package の規約検証（CLAUDE.md）。

``src/jidohub/__init__.py`` が存在すると名前空間が占有され、sibling の jidohub
パッケージ（core / datasets）が import できなくなる。機械的に不在を検証する。
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_jidohub_namespace_has_no_init() -> None:
    offending = REPO_ROOT / "src" / "jidohub" / "__init__.py"
    assert not offending.exists(), (
        "src/jidohub/__init__.py must not exist (PEP 420 namespace package); "
        "its presence breaks imports of sibling jidohub packages"
    )


def test_agents_package_has_init() -> None:
    assert (REPO_ROOT / "src" / "jidohub" / "agents" / "__init__.py").exists()


def test_public_imports_succeed() -> None:
    import jidohub.core  # noqa: F401

    import jidohub.agents  # noqa: F401
    from jidohub.agents import AutoAgent, BaseAgent, StreamingMixin  # noqa: F401
