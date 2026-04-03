"""Approval store for tools that require user confirmation before execution."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Timeout waiting for user approval (seconds)
_APPROVAL_TIMEOUT = 120


class ApprovalStore:
    """Thread-safe store for pending tool approval requests.

    Each pending request is an asyncio.Future keyed by a unique request ID.
    The agent loop creates a future and awaits it; the HTTP endpoint resolves
    it when the user clicks Allow or Deny.
    """

    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future[bool]] = {}

    def create(self, request_id: str) -> asyncio.Future[bool]:
        """Create a pending approval future. Returns the Future to await."""
        loop = asyncio.get_event_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self._pending[request_id] = future
        return future

    def resolve(self, request_id: str, approved: bool) -> bool:
        """Resolve a pending approval. Returns True if the request existed."""
        future = self._pending.pop(request_id, None)
        if future is None or future.done():
            return False
        future.set_result(approved)
        return True

    def cancel(self, request_id: str) -> None:
        """Cancel a pending approval (e.g. on disconnect)."""
        future = self._pending.pop(request_id, None)
        if future is not None and not future.done():
            future.cancel()

    def has_pending(self, request_id: str) -> bool:
        return request_id in self._pending

    async def wait_for_approval(self, request_id: str) -> bool:
        """Wait for user approval with timeout. Returns True if approved."""
        future = self._pending.get(request_id)
        if future is None:
            return False
        try:
            return await asyncio.wait_for(future, timeout=_APPROVAL_TIMEOUT)
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            logger.warning("Approval timed out for request %s", request_id)
            return False
        except asyncio.CancelledError:
            self._pending.pop(request_id, None)
            return False


# Global singleton
_approval_store: ApprovalStore | None = None


def get_approval_store() -> ApprovalStore:
    """Get or create the global approval store."""
    global _approval_store
    if _approval_store is None:
        _approval_store = ApprovalStore()
    return _approval_store
