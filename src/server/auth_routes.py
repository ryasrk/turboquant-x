"""Auth + chat-memory API routes.

/v1/auth/register  POST  { username, password }
/v1/auth/login     POST  { username, password }
/v1/auth/me        GET   (requires token)

/v1/sessions       GET   list sessions
/v1/sessions       POST  create session
/v1/sessions/{id}  PATCH rename session
/v1/sessions/{id}  DELETE delete session

/v1/sessions/{id}/messages  GET  load messages
/v1/sessions/{id}/messages  POST save message
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from src.server.auth import (
    create_token,
    decode_token,
    hash_password,
    verify_password,
)
from src.server.database import (
    add_message,
    create_session,
    create_user,
    delete_session,
    get_messages,
    get_user_by_username,
    list_sessions,
    update_session_title,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Schemas ──────────────────────────────────────────────────────────

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,32}$")


class AuthRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=4, max_length=128)


class TokenResponse(BaseModel):
    token: str
    user_id: str
    username: str


class SessionCreate(BaseModel):
    title: str = Field(default="New Chat", max_length=200)


class SessionPatch(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class MessageCreate(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(default="", max_length=64_000)


# ── Auth dependency ──────────────────────────────────────────────────

def get_current_user(authorization: str | None = Header(default=None)) -> dict:
    """Extract and validate JWT from Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return {"user_id": payload["sub"], "username": payload["username"]}


# ── Auth routes ──────────────────────────────────────────────────────

@router.post("/v1/auth/register", response_model=TokenResponse)
async def register(body: AuthRequest):
    if not _USERNAME_RE.match(body.username):
        raise HTTPException(
            status_code=422,
            detail="Username must be 3-32 chars: letters, digits, underscores only",
        )
    if get_user_by_username(body.username):
        raise HTTPException(status_code=409, detail="Username already taken")

    hashed = hash_password(body.password)
    user = create_user(body.username, hashed)
    token = create_token(user["id"], user["username"])
    return TokenResponse(token=token, user_id=user["id"], username=user["username"])


@router.post("/v1/auth/login", response_model=TokenResponse)
async def login(body: AuthRequest):
    user = get_user_by_username(body.username)
    if not user or not verify_password(body.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_token(user["id"], user["username"])
    return TokenResponse(token=token, user_id=user["id"], username=user["username"])


@router.get("/v1/auth/me")
async def me(user: dict = Depends(get_current_user)):
    return {"user_id": user["user_id"], "username": user["username"]}


# ── Session routes ───────────────────────────────────────────────────

@router.get("/v1/sessions")
async def sessions_list(user: dict = Depends(get_current_user)):
    return list_sessions(user["user_id"])


@router.post("/v1/sessions", status_code=201)
async def session_create(body: SessionCreate, user: dict = Depends(get_current_user)):
    return create_session(user["user_id"], body.title)


@router.patch("/v1/sessions/{session_id}")
async def session_rename(session_id: str, body: SessionPatch, user: dict = Depends(get_current_user)):
    ok = update_session_title(session_id, user["user_id"], body.title)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@router.delete("/v1/sessions/{session_id}")
async def session_delete(session_id: str, user: dict = Depends(get_current_user)):
    ok = delete_session(session_id, user["user_id"])
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


# ── Message routes ───────────────────────────────────────────────────

@router.get("/v1/sessions/{session_id}/messages")
async def messages_list(session_id: str, user: dict = Depends(get_current_user)):
    return get_messages(session_id)


@router.post("/v1/sessions/{session_id}/messages", status_code=201)
async def message_save(session_id: str, body: MessageCreate, user: dict = Depends(get_current_user)):
    return add_message(session_id, body.role, body.content)
