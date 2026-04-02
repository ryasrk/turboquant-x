"""Tests for code_tools: GrepCodeTool, PythonEvalTool, CountLinesTool."""
from __future__ import annotations

import pytest

from src.agent.tools.code_tools import CountLinesTool, GrepCodeTool, PythonEvalTool


@pytest.fixture()
def workspace(tmp_path):
    """Create a small workspace with sample files."""
    (tmp_path / "hello.py").write_text("# greeting\nprint('hello world')\n")
    (tmp_path / "data.txt").write_text("line one\nline two\nline three\n")
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "util.py").write_text("def add(a, b):\n    return a + b\n")
    (sub / "notes.md").write_text("# Notes\nSome notes here.\n")
    return tmp_path


# ── GrepCodeTool ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_grep_code_finds_pattern(workspace):
    tool = GrepCodeTool(workspace_root=str(workspace))
    result = await tool.execute(pattern="hello")
    assert "hello.py" in result
    assert "hello world" in result


@pytest.mark.asyncio
async def test_grep_code_file_glob(workspace):
    tool = GrepCodeTool(workspace_root=str(workspace))
    result = await tool.execute(pattern="line", file_glob="*.txt")
    assert "data.txt" in result
    # .py files should not appear even if they match
    assert "hello.py" not in result


@pytest.mark.asyncio
async def test_grep_code_max_results(workspace):
    # Write a file with many matching lines
    big = "\n".join(f"match line {i}" for i in range(100))
    (workspace / "big.txt").write_text(big)
    tool = GrepCodeTool(workspace_root=str(workspace))
    result = await tool.execute(pattern="match", max_results=5)
    assert result.count("\n") <= 4  # 5 lines = 4 newlines


# ── PythonEvalTool ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_python_eval_basic_math():
    tool = PythonEvalTool()
    assert await tool.execute(expression="1 + 2") == "3"


@pytest.mark.asyncio
async def test_python_eval_math_sqrt():
    tool = PythonEvalTool()
    assert await tool.execute(expression="math.sqrt(16)") == "4.0"


@pytest.mark.asyncio
async def test_python_eval_blocks_import():
    tool = PythonEvalTool()
    result = await tool.execute(expression="__import__('os').getcwd()")
    assert "not allowed" in result.lower() or "error" in result.lower()


@pytest.mark.asyncio
async def test_python_eval_blocks_exec():
    tool = PythonEvalTool()
    result = await tool.execute(expression="exec('x=1')")
    assert "not allowed" in result.lower() or "error" in result.lower()


@pytest.mark.asyncio
async def test_python_eval_blocks_eval():
    tool = PythonEvalTool()
    result = await tool.execute(expression="eval('1+1')")
    assert "not allowed" in result.lower() or "error" in result.lower()


@pytest.mark.asyncio
async def test_python_eval_blocks_open():
    tool = PythonEvalTool()
    result = await tool.execute(expression="open('/etc/passwd')")
    assert "not allowed" in result.lower() or "error" in result.lower()


# ── CountLinesTool ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_count_lines_single_file(workspace):
    tool = CountLinesTool(workspace_root=str(workspace))
    result = await tool.execute(path="hello.py")
    assert "2 lines" in result
    assert "hello.py" in result


@pytest.mark.asyncio
async def test_count_lines_directory_with_glob(workspace):
    tool = CountLinesTool(workspace_root=str(workspace))
    result = await tool.execute(path=".", file_glob="*.py")
    # Should include hello.py (2 lines) and pkg/util.py (2 lines)
    assert "Total:" in result
    assert "4 lines" in result
    # Should NOT include .txt or .md files
    assert "data.txt" not in result
    assert "notes.md" not in result
