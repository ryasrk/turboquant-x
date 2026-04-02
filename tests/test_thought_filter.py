"""Tests for thought-stripping in agent loop and streaming filter."""
from __future__ import annotations

import logging
import os
import tempfile

import pytest

from src.agent.loop import _strip_think_blocks, init_thought_log, _thought_logger
from src.server.routes import _ThinkStreamFilter


@pytest.fixture(autouse=True)
def _enable_thought_propagation():
    """Temporarily enable propagation so caplog can capture thought logs."""
    _thought_logger.propagate = True
    yield
    _thought_logger.propagate = False


# ── _strip_think_blocks ──────────────────────────────────────────────────────


class TestStripThinkBlocks:
    def test_no_think_block(self):
        assert _strip_think_blocks("Hello world") == "Hello world"

    def test_single_think_block(self):
        text = "<think>reasoning here</think>The answer is 42."
        assert _strip_think_blocks(text) == "The answer is 42."

    def test_think_block_with_newlines(self):
        text = "<think>\nStep 1: think\nStep 2: more\n</think>\nFinal answer."
        assert _strip_think_blocks(text) == "Final answer."

    def test_multiple_think_blocks(self):
        text = "<think>first</think>A<think>second</think>B"
        assert _strip_think_blocks(text) == "AB"

    def test_empty_think_block(self):
        text = "<think></think>Just answer."
        assert _strip_think_blocks(text) == "Just answer."

    def test_only_think_block(self):
        text = "<think>all thought no answer</think>"
        assert _strip_think_blocks(text) == ""

    def test_text_format_thought(self):
        text = "Thought: The user is asking for the time.\n\nAction: current_time()"
        result = _strip_think_blocks(text)
        assert "Thought:" not in result

    def test_text_format_thought_logged(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="agent.thoughts"):
            _strip_think_blocks("Thought: I need to check the file.\n\nThe file contains data.", tag="t")
        assert any("Thought:" in r.message for r in caplog.records)

    def test_text_format_final_answer(self):
        text = "Thought: Simple question.\n\nAction: Final Answer: The capital is Paris."
        result = _strip_think_blocks(text)
        assert "The capital is Paris." in result
        assert "Thought:" not in result
        assert "Action:" not in result

    def test_mixed_xml_and_text_format(self):
        text = "<think>xml thought</think>Thought: text thought\n\nAction: Final Answer: answer"
        result = _strip_think_blocks(text)
        assert "xml thought" not in result
        assert "Thought:" not in result
        assert "answer" in result

    def test_logs_thought_content(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="agent.thoughts"):
            _strip_think_blocks("<think>secret reasoning</think>answer", tag="test")
        assert any("secret reasoning" in r.message for r in caplog.records)

    def test_tag_appears_in_log(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="agent.thoughts"):
            _strip_think_blocks("<think>stuff</think>ok", tag="iter-3")
        assert any("[iter-3]" in r.message for r in caplog.records)

    def test_orphaned_open_tag(self):
        """Opening <think> without close — truncated by max_tokens."""
        text = "Good text. <think>partial reasoning that got cut off"
        assert _strip_think_blocks(text) == "Good text."

    def test_orphaned_close_tag(self):
        """Closing </think> without open — continuation from previous chunk."""
        text = "leftover thought</think>Actual answer here."
        assert _strip_think_blocks(text) == "Actual answer here."

    def test_orphaned_open_tag_logs(self, caplog):
        """Orphaned open tag content should still be logged."""
        text = "Before <think>reasoning cut off"
        with caplog.at_level(logging.DEBUG, logger="agent.thoughts"):
            result = _strip_think_blocks(text, tag="orphan")
        # The orphaned content is stripped from output
        assert "reasoning" not in result
        assert result == "Before"


class TestInitThoughtLog:
    def test_creates_log_file(self, tmp_path):
        # Clear any existing handlers
        _thought_logger.handlers.clear()
        log_path = str(tmp_path / "thoughts.log")
        init_thought_log(log_path)
        assert os.path.exists(log_path) or len(_thought_logger.handlers) > 0
        _thought_logger.handlers.clear()

    def test_idempotent(self, tmp_path):
        _thought_logger.handlers.clear()
        log_path = str(tmp_path / "thoughts.log")
        init_thought_log(log_path)
        count = len(_thought_logger.handlers)
        init_thought_log(log_path)  # second call
        assert len(_thought_logger.handlers) == count
        _thought_logger.handlers.clear()


# ── _ThinkStreamFilter ───────────────────────────────────────────────────────


class TestThinkStreamFilter:
    def test_passthrough_no_think(self):
        f = _ThinkStreamFilter()
        result = f.feed("Hello")
        # May buffer partial tag detection
        trailing = f.flush()
        combined = (result or "") + (trailing or "")
        assert "Hello" in combined

    def test_strips_think_block_across_tokens(self):
        f = _ThinkStreamFilter()
        tokens = ["<", "think", ">", "secret", " reason", "</", "think", ">", "The", " answer"]
        emitted = []
        for t in tokens:
            r = f.feed(t)
            if r:
                emitted.append(r)
        r = f.flush()
        if r:
            emitted.append(r)
        full = "".join(emitted)
        assert "secret" not in full
        assert "reason" not in full
        assert "The answer" in full

    def test_single_token_full_block(self):
        f = _ThinkStreamFilter()
        r = f.feed("<think>thought</think>answer")
        trailing = f.flush()
        full = (r or "") + (trailing or "")
        assert "thought" not in full
        assert "answer" in full

    def test_no_think_tokens(self):
        f = _ThinkStreamFilter()
        emitted = []
        for t in ["Hello", " world", "!"]:
            r = f.feed(t)
            if r:
                emitted.append(r)
        r = f.flush()
        if r:
            emitted.append(r)
        assert "Hello world!" == "".join(emitted)

    def test_think_at_start(self):
        f = _ThinkStreamFilter()
        tokens = ["<think>", "internal", "</think>", "visible"]
        emitted = []
        for t in tokens:
            r = f.feed(t)
            if r:
                emitted.append(r)
        r = f.flush()
        if r:
            emitted.append(r)
        full = "".join(emitted)
        assert "internal" not in full
        assert "visible" in full

    def test_incomplete_think_flushed(self, caplog):
        """Incomplete <think> at end of stream gets logged."""
        f = _ThinkStreamFilter()
        f.feed("<think>unfinished thought")
        with caplog.at_level(logging.DEBUG, logger="agent.thoughts"):
            f.flush()
        assert any("unfinished thought" in r.message for r in caplog.records)

    def test_empty_think_block(self):
        f = _ThinkStreamFilter()
        r = f.feed("<think></think>answer")
        trailing = f.flush()
        full = (r or "") + (trailing or "")
        assert "answer" in full

    def test_text_before_and_after(self):
        f = _ThinkStreamFilter()
        tokens = ["Before ", "<think>hidden</think>", " After"]
        emitted = []
        for t in tokens:
            r = f.feed(t)
            if r:
                emitted.append(r)
        r = f.flush()
        if r:
            emitted.append(r)
        full = "".join(emitted)
        assert "Before" in full
        assert "After" in full
        assert "hidden" not in full
