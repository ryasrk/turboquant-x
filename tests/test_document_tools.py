"""Tests for document generation tools (Word, PDF, CSV)."""
from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

import pytest

from src.agent.tools.document_tools import (
    GenerateCsvTool,
    GeneratePdfTool,
    GenerateWordTool,
    _OUTPUT_DIR,
)


@pytest.fixture(autouse=True)
def _clean_output():
    """Ensure a clean output directory for each test."""
    if _OUTPUT_DIR.exists():
        shutil.rmtree(_OUTPUT_DIR)
    yield
    if _OUTPUT_DIR.exists():
        shutil.rmtree(_OUTPUT_DIR)


# ---------------------------------------------------------------------------
# Word
# ---------------------------------------------------------------------------

class TestGenerateWordTool:
    def test_schema(self):
        tool = GenerateWordTool()
        assert tool.name == "generate_word"
        assert "title" in tool.parameters["properties"]
        assert "content" in tool.parameters["properties"]

    @pytest.mark.asyncio
    async def test_generates_docx(self):
        tool = GenerateWordTool()
        result = await tool.execute(
            title="Test Doc",
            content="Hello\n## Heading\nBody text",
            filename="test_word",
        )
        assert "test_word.docx" in result
        assert "[Download test_word.docx]" in result
        assert "/v1/documents/download/test_word.docx" in result
        assert (_OUTPUT_DIR / "test_word.docx").is_file()

    @pytest.mark.asyncio
    async def test_auto_filename(self):
        tool = GenerateWordTool()
        result = await tool.execute(title="Auto", content="Content")
        assert ".docx" in result
        assert "[Download" in result

    @pytest.mark.asyncio
    async def test_sanitises_filename(self):
        tool = GenerateWordTool()
        result = await tool.execute(
            title="T", content="C", filename="../../etc/passwd"
        )
        # Path traversal chars are stripped
        assert "etcpasswd.docx" in result


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

class TestGeneratePdfTool:
    def test_schema(self):
        tool = GeneratePdfTool()
        assert tool.name == "generate_pdf"
        assert "title" in tool.parameters["properties"]

    @pytest.mark.asyncio
    async def test_generates_pdf(self):
        tool = GeneratePdfTool()
        result = await tool.execute(
            title="Test PDF",
            content="Hello\n## Section\n### Sub\nBody",
            filename="test_pdf",
        )
        assert "test_pdf.pdf" in result
        assert "[Download test_pdf.pdf]" in result
        assert "/v1/documents/download/test_pdf.pdf" in result
        assert (_OUTPUT_DIR / "test_pdf.pdf").is_file()
        # Verify it's a real PDF
        header = (_OUTPUT_DIR / "test_pdf.pdf").read_bytes()[:5]
        assert header == b"%PDF-"


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

class TestGenerateCsvTool:
    def test_schema(self):
        tool = GenerateCsvTool()
        assert tool.name == "generate_csv"

    @pytest.mark.asyncio
    async def test_generates_csv_from_headers_rows(self):
        tool = GenerateCsvTool()
        result = await tool.execute(
            headers=["Name", "Age"],
            rows=[["Alice", "30"], ["Bob", "25"]],
            filename="test_csv",
        )
        assert "test_csv.csv" in result
        content = (_OUTPUT_DIR / "test_csv.csv").read_text()
        assert "Name,Age" in content
        assert "Alice,30" in content

    @pytest.mark.asyncio
    async def test_generates_csv_from_raw(self):
        tool = GenerateCsvTool()
        raw = "Col1,Col2\na,b\nc,d"
        result = await tool.execute(raw_csv=raw, filename="raw_test")
        assert "raw_test.csv" in result
        content = (_OUTPUT_DIR / "raw_test.csv").read_text()
        assert content == raw

    @pytest.mark.asyncio
    async def test_error_on_empty(self):
        tool = GenerateCsvTool()
        result = await tool.execute()
        assert "Error" in result
