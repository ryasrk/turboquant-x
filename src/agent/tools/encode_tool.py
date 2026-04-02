"""Encoding/decoding utility tool: base64, URL, hashing."""
from __future__ import annotations

import base64
import hashlib
import urllib.parse
from typing import Any

from src.agent.base import Tool

_MAX_INPUT = 50_000  # 50KB

_OPERATIONS = {
    "base64_encode", "base64_decode",
    "url_encode", "url_decode",
    "sha256", "sha1", "md5",
    "hex_encode", "hex_decode",
}


class EncodeDecodeTool(Tool):
    """Encode, decode, or hash text using various algorithms."""

    @property
    def name(self) -> str:
        return "encode_decode"

    @property
    def description(self) -> str:
        return (
            "Encode, decode, or hash text. Supported operations: "
            "base64_encode, base64_decode, url_encode, url_decode, "
            "sha256, sha1, md5, hex_encode, hex_decode."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "The encoding operation to perform",
                    "enum": list(_OPERATIONS),
                },
                "text": {
                    "type": "string",
                    "description": "The text to process",
                },
            },
            "required": ["operation", "text"],
        }

    async def execute(self, **kwargs: Any) -> str:
        operation: str = kwargs["operation"].strip().lower()
        text: str = kwargs["text"]

        if operation not in _OPERATIONS:
            return f"Error: unknown operation '{operation}'. Supported: {', '.join(sorted(_OPERATIONS))}"
        if len(text) > _MAX_INPUT:
            return f"Error: input too large (>{_MAX_INPUT // 1000}KB)."

        try:
            if operation == "base64_encode":
                return base64.b64encode(text.encode()).decode()
            elif operation == "base64_decode":
                return base64.b64decode(text).decode()
            elif operation == "url_encode":
                return urllib.parse.quote(text, safe="")
            elif operation == "url_decode":
                return urllib.parse.unquote(text)
            elif operation == "sha256":
                return hashlib.sha256(text.encode()).hexdigest()
            elif operation == "sha1":
                return hashlib.sha1(text.encode()).hexdigest()
            elif operation == "md5":
                return hashlib.md5(text.encode()).hexdigest()
            elif operation == "hex_encode":
                return text.encode().hex()
            elif operation == "hex_decode":
                return bytes.fromhex(text).decode()
            else:
                return f"Error: operation '{operation}' not implemented."
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"
