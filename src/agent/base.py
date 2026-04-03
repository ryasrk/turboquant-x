from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

_SCHEMA_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


class Tool(ABC):
    """Abstract base class for agent tools."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> dict: ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str: ...

    @property
    def requires_approval(self) -> bool:
        """Whether this tool requires user approval before execution."""
        return False

    def to_schema(self) -> dict:
        """Return OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate_params(self, params: dict) -> None:
        """Validate that all required parameters are present."""
        required = self.parameters.get("required", [])
        missing = [r for r in required if r not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")

    def cast_params(self, params: dict) -> dict:
        """Cast parameter values to match their JSON Schema types.

        Returns a new dict — never mutates the original.
        """
        properties = self.parameters.get("properties", {})
        casted: dict[str, Any] = {}
        for key, value in params.items():
            schema = properties.get(key)
            if schema is not None:
                target_type = _SCHEMA_TYPE_MAP.get(schema.get("type", ""))
                if target_type is not None:
                    try:
                        value = target_type(value)
                    except (ValueError, TypeError):
                        pass
            casted[key] = value
        return casted
