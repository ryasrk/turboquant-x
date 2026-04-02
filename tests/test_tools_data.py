"""Tests for data tools: JsonQueryTool, CalculateTool, CsvReadTool."""
from __future__ import annotations

import json
import math

import pytest

from src.agent.tools.data_tools import CalculateTool, CsvReadTool, JsonQueryTool


# ---------------------------------------------------------------------------
# JsonQueryTool
# ---------------------------------------------------------------------------

class TestJsonQueryTool:

    @pytest.mark.asyncio
    async def test_loads_full_file(self, tmp_path):
        data = {"users": [{"name": "Alice"}, {"name": "Bob"}]}
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))

        tool = JsonQueryTool(workspace_root=str(tmp_path))
        result = await tool.execute(path="data.json")
        assert json.loads(result) == data

    @pytest.mark.asyncio
    async def test_dot_notation_path(self, tmp_path):
        data = {"a": {"b": {"c": 42}}}
        f = tmp_path / "nested.json"
        f.write_text(json.dumps(data))

        tool = JsonQueryTool(workspace_root=str(tmp_path))
        result = await tool.execute(path="nested.json", query="a.b.c")
        assert json.loads(result) == 42

    @pytest.mark.asyncio
    async def test_array_index(self, tmp_path):
        data = {"items": ["zero", "one", "two"]}
        f = tmp_path / "arr.json"
        f.write_text(json.dumps(data))

        tool = JsonQueryTool(workspace_root=str(tmp_path))
        result = await tool.execute(path="arr.json", query="items[1]")
        assert json.loads(result) == "one"

    @pytest.mark.asyncio
    async def test_file_not_found(self, tmp_path):
        tool = JsonQueryTool(workspace_root=str(tmp_path))
        result = await tool.execute(path="nope.json")
        assert "Error" in result
        assert "file not found" in result

    @pytest.mark.asyncio
    async def test_invalid_query_path(self, tmp_path):
        data = {"a": 1}
        f = tmp_path / "simple.json"
        f.write_text(json.dumps(data))

        tool = JsonQueryTool(workspace_root=str(tmp_path))
        result = await tool.execute(path="simple.json", query="a.b.c")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, tmp_path):
        tool = JsonQueryTool(workspace_root=str(tmp_path))
        result = await tool.execute(path="../../../etc/passwd")
        assert "Error" in result
        assert "escapes workspace" in result


# ---------------------------------------------------------------------------
# CalculateTool
# ---------------------------------------------------------------------------

class TestCalculateTool:

    @pytest.mark.asyncio
    async def test_basic_arithmetic(self):
        tool = CalculateTool()
        assert await tool.execute(expression="2 + 3 * 4") == "14"

    @pytest.mark.asyncio
    async def test_trig_sin(self):
        tool = CalculateTool()
        result = float(await tool.execute(expression="sin(0)"))
        assert result == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_trig_cos(self):
        tool = CalculateTool()
        result = float(await tool.execute(expression="cos(0)"))
        assert result == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_sqrt_and_pi(self):
        tool = CalculateTool()
        result = float(await tool.execute(expression="sqrt(pi)"))
        assert result == pytest.approx(math.sqrt(math.pi))

    @pytest.mark.asyncio
    async def test_blocks_dangerous_nodes(self):
        tool = CalculateTool()
        # import expression should fail
        result = await tool.execute(expression="__import__('os').system('id')")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_blocks_lambda(self):
        tool = CalculateTool()
        result = await tool.execute(expression="(lambda: 1)()")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_division_by_zero(self):
        tool = CalculateTool()
        result = await tool.execute(expression="1/0")
        assert "Error" in result
        assert "division by zero" in result


# ---------------------------------------------------------------------------
# CsvReadTool
# ---------------------------------------------------------------------------

class TestCsvReadTool:

    def _write_csv(self, tmp_path, filename, header, rows):
        f = tmp_path / filename
        lines = [",".join(header)]
        for row in rows:
            lines.append(",".join(str(v) for v in row))
        f.write_text("\n".join(lines))
        return f

    @pytest.mark.asyncio
    async def test_read_full_file(self, tmp_path):
        self._write_csv(
            tmp_path, "data.csv",
            ["name", "age"],
            [["Alice", "30"], ["Bob", "25"]],
        )
        tool = CsvReadTool(workspace_root=str(tmp_path))
        result = await tool.execute(path="data.csv")
        assert "Alice" in result
        assert "Bob" in result
        assert "name" in result
        assert "age" in result

    @pytest.mark.asyncio
    async def test_column_filter(self, tmp_path):
        self._write_csv(
            tmp_path, "data.csv",
            ["name", "age", "city"],
            [["Alice", "30", "NYC"], ["Bob", "25", "LA"]],
        )
        tool = CsvReadTool(workspace_root=str(tmp_path))
        result = await tool.execute(path="data.csv", columns=["name", "city"])
        assert "name" in result
        assert "city" in result
        # age column should not appear as a header row entry
        lines = result.strip().split("\n")
        assert "age" not in lines[0]

    @pytest.mark.asyncio
    async def test_row_filter(self, tmp_path):
        self._write_csv(
            tmp_path, "data.csv",
            ["name", "age"],
            [["Alice", "30"], ["Bob", "25"], ["Carol", "30"]],
        )
        tool = CsvReadTool(workspace_root=str(tmp_path))
        result = await tool.execute(
            path="data.csv", filter_column="age", filter_value="30",
        )
        assert "Alice" in result
        assert "Carol" in result
        assert "Bob" not in result

    @pytest.mark.asyncio
    async def test_max_rows_truncation(self, tmp_path):
        rows = [[f"user{i}", str(i)] for i in range(100)]
        self._write_csv(tmp_path, "big.csv", ["name", "num"], rows)

        tool = CsvReadTool(workspace_root=str(tmp_path))
        result = await tool.execute(path="big.csv", max_rows=5)
        assert "truncated" in result
        assert "100 total rows" in result
        assert "showing 5" in result

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, tmp_path):
        tool = CsvReadTool(workspace_root=str(tmp_path))
        result = await tool.execute(path="../../etc/shadow")
        assert "Error" in result
        assert "escapes workspace" in result
