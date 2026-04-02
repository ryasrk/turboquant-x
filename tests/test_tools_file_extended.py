"""Tests for FindFilesTool and ReplaceInFileTool."""
from __future__ import annotations

import pytest

from src.agent.tools.file_tools import FindFilesTool, ReplaceInFileTool


@pytest.fixture()
def workspace(tmp_path):
    """Create a small workspace tree for testing."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')\n")
    (tmp_path / "src" / "utils.py").write_text("x = 1\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "readme.md").write_text("# README\n")
    (tmp_path / "data.json").write_text('{"key": "value"}\n')
    # dirs that should be skipped
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.js").write_text("")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("")
    return tmp_path


# ── FindFilesTool: glob ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_find_files_glob(workspace):
    tool = FindFilesTool(workspace_root=str(workspace))
    result = await tool.execute(pattern="*.py")
    assert "main.py" in result
    assert "utils.py" in result
    # skipped dirs should not appear
    assert "node_modules" not in result
    assert ".git" not in result


# ── FindFilesTool: regex ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_find_files_regex(workspace):
    tool = FindFilesTool(workspace_root=str(workspace))
    result = await tool.execute(pattern=r"\.py$", regex=True)
    assert "main.py" in result
    assert "utils.py" in result
    assert "readme.md" not in result


# ── FindFilesTool: max_results cap ───────────────────────────────────


@pytest.mark.asyncio
async def test_find_files_max_results(workspace):
    # Create enough files to trigger truncation
    lots = workspace / "many"
    lots.mkdir()
    for i in range(10):
        (lots / f"f{i}.txt").write_text("")

    tool = FindFilesTool(workspace_root=str(workspace))
    result = await tool.execute(pattern="*.txt", max_results=3)
    assert "[truncated" in result
    # Should only have 3 file lines (before the truncated note)
    file_lines = [l for l in result.splitlines() if not l.startswith("[")]
    assert len(file_lines) == 3


# ── ReplaceInFileTool: single replacement ────────────────────────────


@pytest.mark.asyncio
async def test_replace_single(workspace):
    (workspace / "target.txt").write_text("aaa bbb aaa")
    tool = ReplaceInFileTool(workspace_root=str(workspace))
    result = await tool.execute(path="target.txt", old_text="aaa", new_text="ccc")
    assert "Replaced 1 occurrence(s)" in result
    content = (workspace / "target.txt").read_text()
    assert content == "ccc bbb aaa"


# ── ReplaceInFileTool: multiple replacements ─────────────────────────


@pytest.mark.asyncio
async def test_replace_multiple(workspace):
    (workspace / "target.txt").write_text("aaa bbb aaa")
    tool = ReplaceInFileTool(workspace_root=str(workspace))
    result = await tool.execute(
        path="target.txt", old_text="aaa", new_text="ccc", max_replacements=0
    )
    assert "Replaced 2 occurrence(s)" in result
    content = (workspace / "target.txt").read_text()
    assert content == "ccc bbb ccc"


# ── ReplaceInFileTool: no match ──────────────────────────────────────


@pytest.mark.asyncio
async def test_replace_no_match(workspace):
    (workspace / "target.txt").write_text("hello world")
    tool = ReplaceInFileTool(workspace_root=str(workspace))
    result = await tool.execute(path="target.txt", old_text="zzz", new_text="yyy")
    assert "No matches found" in result


# ── ReplaceInFileTool: path traversal blocked ────────────────────────


@pytest.mark.asyncio
async def test_replace_path_traversal(workspace):
    tool = ReplaceInFileTool(workspace_root=str(workspace))
    result = await tool.execute(
        path="../../../etc/passwd", old_text="root", new_text="hacked"
    )
    assert "Error" in result
    assert "escapes workspace" in result.lower() or "Path escapes" in result
