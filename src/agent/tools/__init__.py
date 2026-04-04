"""Built-in tools for agent system."""
from __future__ import annotations

from src.agent.tools.file_tools import (
    ListDirTool, ReadFileTool, WriteFileTool, FindFilesTool, ReplaceInFileTool,
)
from src.agent.tools.shell_tool import ExecTool
from src.agent.tools.web_search_tool import WebSearchTool
from src.agent.tools.web_tools import FetchWebpageTool, HttpRequestTool
from src.agent.tools.system_tools import GetEnvTool, CurrentTimeTool
from src.agent.tools.code_tools import GrepCodeTool, PythonEvalTool, CountLinesTool
from src.agent.tools.data_tools import JsonQueryTool, CalculateTool, CsvReadTool
from src.agent.tools.rag_tools import ReadPdfTool, IndexDocumentTool, SearchDocumentTool
from src.agent.tools.memory_tools import SaveNoteTool, RecallNoteTool, DeleteNoteTool
from src.agent.tools.sql_tool import SqlQueryTool
from src.agent.tools.sysinfo_tool import SystemInfoTool
from src.agent.tools.diff_tool import DiffFilesTool
from src.agent.tools.encode_tool import EncodeDecodeTool
from src.agent.tools.terminal_tool import TerminalTool
from src.agent.tools.mcp_bridge_tool import McpBridgeTool
from src.agent.tools.document_tools import GenerateWordTool, GeneratePdfTool, GenerateCsvTool

__all__ = [
    # File System
    "ReadFileTool", "WriteFileTool", "ListDirTool", "FindFilesTool", "ReplaceInFileTool",
    # System
    "ExecTool",
    # Web
    "WebSearchTool", "FetchWebpageTool", "HttpRequestTool",
    # System Utils
    "GetEnvTool", "CurrentTimeTool", "SystemInfoTool",
    # Code Analysis
    "GrepCodeTool", "PythonEvalTool", "CountLinesTool",
    # Data & Math
    "JsonQueryTool", "CalculateTool", "CsvReadTool",
    # RAG / Document
    "ReadPdfTool", "IndexDocumentTool", "SearchDocumentTool",
    # Memory
    "SaveNoteTool", "RecallNoteTool", "DeleteNoteTool",
    # Database
    "SqlQueryTool",
    # Utility
    "DiffFilesTool", "EncodeDecodeTool",
    # Terminal (requires approval)
    "TerminalTool",
    # MCP Bridge
    "McpBridgeTool",
    # Document Generation
    "GenerateWordTool", "GeneratePdfTool", "GenerateCsvTool",
]
