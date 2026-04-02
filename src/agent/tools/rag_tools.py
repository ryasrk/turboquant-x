"""PageIndex-inspired RAG tools — PDF reading, tree indexing, and reasoning-based retrieval."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from src.agent.base import Tool

MAX_PAGE_CHARS = 8_000
MAX_RESULT_CHARS = 16_000

# Directories skipped when resolving paths
_INDEXES_DIR = "data/document_indexes"


def _resolve_safe(path: str, workspace_root: str) -> Path:
    """Resolve *path* inside *workspace_root* (blocks traversal)."""
    root = Path(workspace_root).resolve()
    resolved = (root / path).resolve()
    if not (resolved == root or str(resolved).startswith(str(root) + os.sep)):
        raise PermissionError(f"Path escapes workspace root: {path}")
    return resolved


def _index_dir(workspace_root: str) -> Path:
    """Return (and create) the indexes directory."""
    d = Path(workspace_root) / _INDEXES_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index_path_for(pdf_path: Path, workspace_root: str) -> Path:
    """Deterministic index file path based on PDF content hash."""
    h = hashlib.sha256(str(pdf_path.resolve()).encode()).hexdigest()[:16]
    return _index_dir(workspace_root) / f"{pdf_path.stem}_{h}.json"


# ---------------------------------------------------------------------------
# 1. ReadPdfTool
# ---------------------------------------------------------------------------

class ReadPdfTool(Tool):
    """Extract text from a PDF file, optionally a page range."""

    def __init__(self, workspace_root: str | None = None) -> None:
        self._workspace_root = workspace_root or os.getcwd()

    @property
    def name(self) -> str:
        return "read_pdf"

    @property
    def description(self) -> str:
        return (
            "Extract text from a PDF file. "
            "Optionally specify a page range (1-based). "
            "Returns text content with page markers."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the PDF file"},
                "start_page": {
                    "type": "integer",
                    "description": "Start page number (1-based, optional)",
                },
                "end_page": {
                    "type": "integer",
                    "description": "End page number (1-based, inclusive, optional)",
                },
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        path_str: str = kwargs["path"]
        start_page: int | None = kwargs.get("start_page")
        end_page: int | None = kwargs.get("end_page")

        try:
            resolved = _resolve_safe(path_str, self._workspace_root)
        except PermissionError as exc:
            return f"Error: {exc}"

        def _read() -> str:
            try:
                import fitz  # pymupdf
            except ImportError:
                return "Error: pymupdf not installed (pip install pymupdf)"

            if not resolved.exists():
                return f"Error: file not found – {path_str}"
            if resolved.suffix.lower() != ".pdf":
                return f"Error: not a PDF file – {path_str}"

            doc = fitz.open(str(resolved))
            total = len(doc)

            s = max((start_page or 1) - 1, 0)
            e = min(end_page or total, total)

            parts: list[str] = []
            chars = 0
            for i in range(s, e):
                page = doc[i]
                text = page.get_text("text")
                parts.append(f"--- Page {i + 1}/{total} ---\n{text}")
                chars += len(text)
                if chars > MAX_PAGE_CHARS * 4:
                    parts.append(f"\n[Truncated at page {i + 1}. Total pages: {total}]")
                    break
            doc.close()

            result = "\n".join(parts)
            if len(result) > MAX_RESULT_CHARS:
                result = result[:MAX_RESULT_CHARS] + "\n... [truncated]"
            return result

        try:
            return await asyncio.to_thread(_read)
        except Exception as exc:
            return f"Error reading PDF: {exc}"


# ---------------------------------------------------------------------------
# 2. IndexDocumentTool
# ---------------------------------------------------------------------------

class IndexDocumentTool(Tool):
    """Build a hierarchical tree index from a document using LLM reasoning.

    The index is a JSON tree of sections with titles, summaries, and page ranges.
    Saved to data/document_indexes/ for persistent reuse.
    """

    def __init__(
        self,
        workspace_root: str | None = None,
        llm_fn: Any = None,
    ) -> None:
        self._workspace_root = workspace_root or os.getcwd()
        self._llm_fn = llm_fn  # async callable(messages) -> str

    @property
    def name(self) -> str:
        return "index_document"

    @property
    def description(self) -> str:
        return (
            "Build a hierarchical tree index from a PDF document. "
            "Creates a PageIndex-style table of contents with titles, summaries, "
            "and page ranges. Saved for later retrieval. Returns the index JSON."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the PDF file"},
                "force": {
                    "type": "boolean",
                    "description": "Re-index even if an index already exists (default false)",
                },
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        path_str: str = kwargs["path"]
        force: bool = kwargs.get("force", False)

        try:
            resolved = _resolve_safe(path_str, self._workspace_root)
        except PermissionError as exc:
            return f"Error: {exc}"

        if not resolved.exists():
            return f"Error: file not found – {path_str}"

        idx_path = _index_path_for(resolved, self._workspace_root)

        # Return cached index unless force
        if idx_path.exists() and not force:
            return f"Index already exists: {idx_path.relative_to(self._workspace_root)}\n\n" + idx_path.read_text()

        # Extract all page text
        pages = await asyncio.to_thread(self._extract_pages, resolved)
        if isinstance(pages, str) and pages.startswith("Error"):
            return pages

        # Build the tree index using LLM
        index = await self._build_tree_index(pages, resolved.name)

        # Save index
        idx_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))

        result = json.dumps(index, indent=2, ensure_ascii=False)
        if len(result) > MAX_RESULT_CHARS:
            result = result[:MAX_RESULT_CHARS] + "\n... [truncated]"
        return f"Index saved to: {idx_path.relative_to(self._workspace_root)}\n\n{result}"

    @staticmethod
    def _extract_pages(pdf_path: Path) -> list[dict[str, Any]] | str:
        """Extract text from all pages."""
        try:
            import fitz
        except ImportError:
            return "Error: pymupdf not installed"

        doc = fitz.open(str(pdf_path))
        pages = []
        for i in range(len(doc)):
            text = doc[i].get_text("text").strip()
            pages.append({"page": i + 1, "text": text})
        doc.close()
        return pages

    async def _build_tree_index(
        self, pages: list[dict], filename: str,
    ) -> dict:
        """Build hierarchical index. Uses LLM if available, else heuristic."""
        if self._llm_fn:
            return await self._build_with_llm(pages, filename)
        return self._build_heuristic(pages, filename)

    async def _build_with_llm(
        self, pages: list[dict], filename: str,
    ) -> dict:
        """Use the LLM to generate a structured table of contents."""
        # Send batches of pages to LLM for section identification
        batch_size = 10
        all_sections: list[dict] = []

        for i in range(0, len(pages), batch_size):
            batch = pages[i : i + batch_size]
            pages_text = "\n\n".join(
                f"[Page {p['page']}]\n{p['text'][:2000]}" for p in batch
            )

            prompt = (
                f"Analyze these pages from '{filename}' and identify the main sections. "
                "For each section, provide a JSON array of objects with: "
                '"title", "start_page", "end_page", "summary" (1-2 sentences). '
                "Only output the JSON array, nothing else.\n\n"
                f"{pages_text}"
            )

            try:
                response = await self._llm_fn([
                    {"role": "system", "content": "You are a document analyst. Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ])
                # Parse JSON from response
                sections = self._parse_json_response(response)
                all_sections.extend(sections)
            except Exception:
                # Fall back to heuristic for this batch
                for p in batch:
                    all_sections.append({
                        "title": f"Page {p['page']}",
                        "start_page": p["page"],
                        "end_page": p["page"],
                        "summary": p["text"][:200],
                    })

        return self._assemble_tree(all_sections, filename, len(pages))

    @staticmethod
    def _parse_json_response(text: str) -> list[dict]:
        """Extract JSON array from LLM response text."""
        # Try to find JSON array in the response
        text = text.strip()
        # Remove markdown code fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        # Find first [ and last ]
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        return []

    @staticmethod
    def _build_heuristic(
        pages: list[dict], filename: str,
    ) -> dict:
        """Fallback: build index from text structure (headings, page breaks)."""
        sections: list[dict] = []
        current_section: dict | None = None

        for page in pages:
            text = page["text"]
            lines = text.split("\n")
            # Detect potential section headings (short lines, often uppercase or title case)
            heading = None
            for line in lines[:5]:  # Check first 5 lines of page
                line = line.strip()
                if 3 < len(line) < 100 and (line.isupper() or line.istitle()):
                    heading = line
                    break

            if heading and (not current_section or heading != current_section.get("title")):
                if current_section:
                    current_section["end_page"] = page["page"] - 1
                    sections.append(current_section)
                current_section = {
                    "title": heading,
                    "start_page": page["page"],
                    "end_page": page["page"],
                    "summary": text[:200].replace("\n", " "),
                }
            elif current_section:
                current_section["end_page"] = page["page"]
            else:
                current_section = {
                    "title": f"Section starting page {page['page']}",
                    "start_page": page["page"],
                    "end_page": page["page"],
                    "summary": text[:200].replace("\n", " "),
                }

        if current_section:
            sections.append(current_section)

        return {
            "filename": filename,
            "total_pages": len(pages),
            "sections": sections,
            "index_type": "heuristic",
        }

    @staticmethod
    def _assemble_tree(
        sections: list[dict], filename: str, total_pages: int,
    ) -> dict:
        """Assemble flat sections into a tree structure."""
        # Merge overlapping sections and build hierarchy
        merged: list[dict] = []
        for s in sections:
            if merged and s.get("start_page", 0) <= merged[-1].get("end_page", 0):
                # Overlap — make child of previous
                if "children" not in merged[-1]:
                    merged[-1]["children"] = []
                merged[-1]["children"].append(s)
                merged[-1]["end_page"] = max(
                    merged[-1].get("end_page", 0), s.get("end_page", 0)
                )
            else:
                merged.append(s)

        return {
            "filename": filename,
            "total_pages": total_pages,
            "sections": merged,
            "index_type": "llm" if len(sections) > 0 else "empty",
        }


# ---------------------------------------------------------------------------
# 3. SearchDocumentTool
# ---------------------------------------------------------------------------

class SearchDocumentTool(Tool):
    """Reasoning-based document search using the tree index.

    Traverses the document tree top-down, expanding only relevant branches,
    then returns the matching page text.
    """

    def __init__(
        self,
        workspace_root: str | None = None,
        llm_fn: Any = None,
    ) -> None:
        self._workspace_root = workspace_root or os.getcwd()
        self._llm_fn = llm_fn

    @property
    def name(self) -> str:
        return "search_document"

    @property
    def description(self) -> str:
        return (
            "Search a previously indexed document using reasoning-based retrieval. "
            "Navigates the document tree to find the most relevant sections for a query. "
            "Returns matching page text with references."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the PDF file (must be indexed first)"},
                "query": {"type": "string", "description": "What to search for in the document"},
                "max_pages": {
                    "type": "integer",
                    "description": "Maximum pages to return (default 3)",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["path", "query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        path_str: str = kwargs["path"]
        query: str = kwargs["query"]
        max_pages: int = min(kwargs.get("max_pages", 3), 10)

        try:
            resolved = _resolve_safe(path_str, self._workspace_root)
        except PermissionError as exc:
            return f"Error: {exc}"

        if not resolved.exists():
            return f"Error: file not found – {path_str}"

        idx_path = _index_path_for(resolved, self._workspace_root)
        if not idx_path.exists():
            return (
                f"Error: no index found for {path_str}. "
                "Use index_document first to create an index."
            )

        # Load index
        try:
            index = json.loads(idx_path.read_text())
        except json.JSONDecodeError:
            return "Error: corrupted index file"

        # Find relevant sections
        relevant_sections = await self._find_relevant_sections(
            index, query, max_pages
        )

        if not relevant_sections:
            return "No relevant sections found for the query."

        # Extract page text for relevant sections
        page_ranges = set()
        for section in relevant_sections:
            start = section.get("start_page", 1)
            end = section.get("end_page", start)
            for p in range(start, min(end + 1, start + max_pages)):
                page_ranges.add(p)

        pages_text = await asyncio.to_thread(
            self._extract_specific_pages, resolved, sorted(page_ranges)
        )

        # Build result with section context
        parts: list[str] = []
        parts.append(f"Document: {index.get('filename', path_str)}")
        parts.append(f"Query: {query}")
        parts.append(f"Relevant sections ({len(relevant_sections)}):")
        for s in relevant_sections:
            parts.append(
                f"  • {s.get('title', 'Untitled')} "
                f"(pages {s.get('start_page', '?')}-{s.get('end_page', '?')}): "
                f"{s.get('summary', '')}"
            )
        parts.append("\n--- Extracted Content ---")
        parts.append(pages_text)

        result = "\n".join(parts)
        if len(result) > MAX_RESULT_CHARS:
            result = result[:MAX_RESULT_CHARS] + "\n... [truncated]"
        return result

    async def _find_relevant_sections(
        self, index: dict, query: str, max_results: int,
    ) -> list[dict]:
        """Find relevant sections using LLM reasoning or keyword matching."""
        sections = index.get("sections", [])
        if not sections:
            return []

        if self._llm_fn:
            return await self._find_with_llm(sections, query, max_results)
        return self._find_with_keywords(sections, query, max_results)

    async def _find_with_llm(
        self, sections: list[dict], query: str, max_results: int,
    ) -> list[dict]:
        """Use LLM to reason about which sections are relevant."""
        sections_desc = "\n".join(
            f"{i}. {s.get('title', 'Untitled')} (pages {s.get('start_page', '?')}-{s.get('end_page', '?')}): {s.get('summary', '')}"
            for i, s in enumerate(sections)
        )

        prompt = (
            f"Given this document's table of contents:\n\n{sections_desc}\n\n"
            f"Query: {query}\n\n"
            f"Which sections (by index number) are most relevant? "
            f"Return a JSON array of at most {max_results} index numbers, "
            f"ordered by relevance. Only output the JSON array."
        )

        try:
            response = await self._llm_fn([
                {"role": "system", "content": "You are a document retrieval expert. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ])
            text = response.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                indices = json.loads(text[start : end + 1])
                return [sections[i] for i in indices if isinstance(i, int) and 0 <= i < len(sections)]
        except Exception:
            pass

        # Fallback to keyword matching
        return self._find_with_keywords(sections, query, max_results)

    @staticmethod
    def _find_with_keywords(
        sections: list[dict], query: str, max_results: int,
    ) -> list[dict]:
        """Simple keyword-based relevance scoring."""
        query_words = set(query.lower().split())

        scored: list[tuple[float, dict]] = []
        for section in sections:
            text = (
                section.get("title", "") + " " + section.get("summary", "")
            ).lower()
            section_words = set(text.split())
            # Jaccard-like score
            overlap = len(query_words & section_words)
            if overlap > 0:
                score = overlap / len(query_words)
                scored.append((score, section))

            # Also check children
            for child in section.get("children", []):
                child_text = (
                    child.get("title", "") + " " + child.get("summary", "")
                ).lower()
                child_words = set(child_text.split())
                child_overlap = len(query_words & child_words)
                if child_overlap > 0:
                    child_score = child_overlap / len(query_words)
                    scored.append((child_score, child))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:max_results]]

    @staticmethod
    def _extract_specific_pages(pdf_path: Path, page_numbers: list[int]) -> str:
        """Extract text from specific pages of a PDF."""
        try:
            import fitz
        except ImportError:
            return "Error: pymupdf not installed"

        doc = fitz.open(str(pdf_path))
        total = len(doc)
        parts: list[str] = []
        chars = 0

        for pn in page_numbers:
            if 1 <= pn <= total:
                text = doc[pn - 1].get_text("text")
                parts.append(f"--- Page {pn}/{total} ---\n{text}")
                chars += len(text)
                if chars > MAX_PAGE_CHARS * 3:
                    parts.append("[Remaining pages truncated]")
                    break
        doc.close()
        return "\n".join(parts) if parts else "No pages extracted."
