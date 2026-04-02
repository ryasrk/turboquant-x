"""Tests for RAG tools — read_pdf, index_document, search_document."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import pytest_asyncio  # noqa: F401  — ensures plugin is loaded

from src.agent.tools.rag_tools import (
    IndexDocumentTool,
    ReadPdfTool,
    SearchDocumentTool,
    _index_path_for,
    _resolve_safe,
)


# ---------------------------------------------------------------------------
# Helper: create a minimal valid PDF using pymupdf
# ---------------------------------------------------------------------------

def _make_pdf(path: Path, pages: list[str]) -> None:
    """Create a minimal PDF with given page texts."""
    import fitz

    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


# ---------------------------------------------------------------------------
# ReadPdfTool
# ---------------------------------------------------------------------------

class TestReadPdfTool:
    @pytest.mark.asyncio
    async def test_read_full_pdf(self, tmp_path: Path) -> None:
        pdf = tmp_path / "test.pdf"
        _make_pdf(pdf, ["Page one content", "Page two content"])
        tool = ReadPdfTool(workspace_root=str(tmp_path))
        result = await tool.execute(path="test.pdf")
        assert "Page 1/2" in result
        assert "Page one content" in result
        assert "Page 2/2" in result

    @pytest.mark.asyncio
    async def test_read_page_range(self, tmp_path: Path) -> None:
        pdf = tmp_path / "test.pdf"
        _make_pdf(pdf, ["Page A", "Page B", "Page C"])
        tool = ReadPdfTool(workspace_root=str(tmp_path))
        result = await tool.execute(path="test.pdf", start_page=2, end_page=2)
        assert "Page 2/3" in result
        assert "Page B" in result
        assert "Page A" not in result

    @pytest.mark.asyncio
    async def test_read_missing_file(self, tmp_path: Path) -> None:
        tool = ReadPdfTool(workspace_root=str(tmp_path))
        result = await tool.execute(path="nope.pdf")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_read_non_pdf(self, tmp_path: Path) -> None:
        txt = tmp_path / "test.txt"
        txt.write_text("not a pdf")
        tool = ReadPdfTool(workspace_root=str(tmp_path))
        result = await tool.execute(path="test.txt")
        assert "not a PDF" in result

    @pytest.mark.asyncio
    async def test_path_traversal(self, tmp_path: Path) -> None:
        tool = ReadPdfTool(workspace_root=str(tmp_path))
        result = await tool.execute(path="../../../etc/passwd")
        assert "Error" in result


# ---------------------------------------------------------------------------
# IndexDocumentTool (heuristic mode — no LLM)
# ---------------------------------------------------------------------------

class TestIndexDocumentTool:
    @pytest.mark.asyncio
    async def test_index_creates_file(self, tmp_path: Path) -> None:
        pdf = tmp_path / "report.pdf"
        _make_pdf(pdf, ["Introduction\nThis is the intro.", "Methods\nDescribing methods."])
        tool = IndexDocumentTool(workspace_root=str(tmp_path))
        result = await tool.execute(path="report.pdf")
        assert "Index saved" in result
        # Verify index file exists
        idx_dir = tmp_path / "data" / "document_indexes"
        assert idx_dir.exists()
        files = list(idx_dir.glob("*.json"))
        assert len(files) == 1
        index = json.loads(files[0].read_text())
        assert index["total_pages"] == 2
        assert len(index["sections"]) > 0

    @pytest.mark.asyncio
    async def test_index_cached(self, tmp_path: Path) -> None:
        pdf = tmp_path / "report.pdf"
        _make_pdf(pdf, ["Content"])
        tool = IndexDocumentTool(workspace_root=str(tmp_path))
        # First index
        await tool.execute(path="report.pdf")
        # Second call should return cached
        result = await tool.execute(path="report.pdf")
        assert "already exists" in result

    @pytest.mark.asyncio
    async def test_index_force_reindex(self, tmp_path: Path) -> None:
        pdf = tmp_path / "report.pdf"
        _make_pdf(pdf, ["Content"])
        tool = IndexDocumentTool(workspace_root=str(tmp_path))
        await tool.execute(path="report.pdf")
        result = await tool.execute(path="report.pdf", force=True)
        assert "Index saved" in result

    @pytest.mark.asyncio
    async def test_index_missing_file(self, tmp_path: Path) -> None:
        tool = IndexDocumentTool(workspace_root=str(tmp_path))
        result = await tool.execute(path="gone.pdf")
        assert "Error" in result


# ---------------------------------------------------------------------------
# SearchDocumentTool (keyword mode — no LLM)
# ---------------------------------------------------------------------------

class TestSearchDocumentTool:
    @pytest.mark.asyncio
    async def test_search_finds_section(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        _make_pdf(pdf, [
            "Financial Overview\nRevenue grew by 15% year over year.",
            "Technical Details\nThe system uses machine learning.",
        ])
        # Index first
        idx_tool = IndexDocumentTool(workspace_root=str(tmp_path))
        await idx_tool.execute(path="doc.pdf")
        # Search
        search_tool = SearchDocumentTool(workspace_root=str(tmp_path))
        result = await search_tool.execute(path="doc.pdf", query="revenue financial")
        assert "Financial" in result or "Page 1" in result

    @pytest.mark.asyncio
    async def test_search_no_index(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        _make_pdf(pdf, ["Content"])
        search_tool = SearchDocumentTool(workspace_root=str(tmp_path))
        result = await search_tool.execute(path="doc.pdf", query="test")
        assert "no index found" in result

    @pytest.mark.asyncio
    async def test_search_no_results(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        _make_pdf(pdf, ["Hello world"])
        idx_tool = IndexDocumentTool(workspace_root=str(tmp_path))
        await idx_tool.execute(path="doc.pdf")
        search_tool = SearchDocumentTool(workspace_root=str(tmp_path))
        result = await search_tool.execute(path="doc.pdf", query="xyznonexistent123")
        assert "No relevant sections" in result

    @pytest.mark.asyncio
    async def test_search_path_traversal(self, tmp_path: Path) -> None:
        search_tool = SearchDocumentTool(workspace_root=str(tmp_path))
        result = await search_tool.execute(path="../../etc/passwd", query="root")
        assert "Error" in result


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestRagUtils:
    def test_resolve_safe_blocks_traversal(self, tmp_path: Path) -> None:
        with pytest.raises(PermissionError):
            _resolve_safe("../../etc/passwd", str(tmp_path))

    def test_index_path_deterministic(self, tmp_path: Path) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.touch()
        p1 = _index_path_for(pdf, str(tmp_path))
        p2 = _index_path_for(pdf, str(tmp_path))
        assert p1 == p2
        assert p1.suffix == ".json"
