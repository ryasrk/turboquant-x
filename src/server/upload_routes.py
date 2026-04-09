"""File upload API routes for TurboQuant-X.

Endpoints:
  POST   /v1/upload                    - Upload files to session
  GET    /v1/attachments/{id}          - Retrieve uploaded file  
  DELETE /v1/attachments/{id}          - Delete uploaded file
  
Background task:
  - cleanup_orphan_attachments()       - Remove orphaned files (no message_id)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .auth_routes import get_current_user
from . import database
from .file_validation import validate_file
from .text_extraction import extract_text

logger = logging.getLogger(__name__)

router = APIRouter()

# Upload directory configuration
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'uploads')

# Rate limiting storage (in-memory)
# Structure: {user_id: [(timestamp, count), ...]}
_rate_limit_storage: Dict[str, List[tuple[float, int]]] = defaultdict(list)
RATE_LIMIT_MAX_UPLOADS = 10
RATE_LIMIT_WINDOW_SECONDS = 60


# ── Schemas ──────────────────────────────────────────────────────────

class AttachmentResponse(BaseModel):
    """Response schema for uploaded attachment."""
    id: str
    original_name: str
    mime_type: str
    size_bytes: int
    type: str  # 'image' | 'document'


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str | None = None


# ── Utility Functions ────────────────────────────────────────────────

def _ensure_upload_dir() -> None:
    """Create upload directory if it doesn't exist."""
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)


def _get_file_type(mime_type: str) -> str:
    """Determine file type from MIME type."""
    if mime_type.startswith('image/'):
        return 'image'
    return 'document'


def _generate_file_hash(content: bytes) -> str:
    """Generate SHA-256 hash of file content."""
    return hashlib.sha256(content).hexdigest()


def _check_rate_limit(user_id: str) -> bool:
    """Check if user has exceeded rate limit. Returns True if allowed."""
    now = time.time()
    user_uploads = _rate_limit_storage[user_id]
    
    # Clean old entries outside window
    cutoff_time = now - RATE_LIMIT_WINDOW_SECONDS
    user_uploads[:] = [(timestamp, count) for timestamp, count in user_uploads if timestamp > cutoff_time]
    
    # Count uploads in current window
    total_uploads = sum(count for _, count in user_uploads)
    
    if total_uploads >= RATE_LIMIT_MAX_UPLOADS:
        return False
    
    # Add current upload
    user_uploads.append((now, 1))
    return True


def _verify_session_ownership(session_id: str, user_id: str) -> bool:
    """Verify that the session belongs to the requesting user."""
    conn = database.get_connection()
    try:
        row = conn.execute(
            "SELECT user_id FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row is not None and row["user_id"] == user_id
    finally:
        conn.close()


def _verify_attachment_ownership(attachment_id: str, user_id: str) -> bool:
    """Verify that the attachment belongs to the requesting user."""
    attachment = database.get_attachment(attachment_id)
    if not attachment:
        return False
    
    session_id = attachment.get('session_id')
    if not session_id:
        return False
    
    return _verify_session_ownership(session_id, user_id)


# ── Upload Route ─────────────────────────────────────────────────────

@router.post("/v1/upload", response_model=AttachmentResponse, status_code=201)
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    user: dict = Depends(get_current_user)
) -> AttachmentResponse:
    """Upload a file to a session.
    
    Args:
        file: The uploaded file (multipart/form-data)
        session_id: Target session ID (form field)
        user: Current user from JWT token
        
    Returns:
        AttachmentResponse with file metadata
        
    Raises:
        HTTPException: 401 (unauthorized), 403 (forbidden), 413 (file too large),
                      415 (unsupported media type), 422 (validation failed)
    """
    user_id = user["user_id"]
    
    # Rate limiting check
    if not _check_rate_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_MAX_UPLOADS} uploads per {RATE_LIMIT_WINDOW_SECONDS} seconds."
        )
    
    # Verify session ownership
    if not _verify_session_ownership(session_id, user_id):
        raise HTTPException(status_code=403, detail="Session not found or access denied")
    
    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read upload file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read file content")
    
    if not content:
        raise HTTPException(status_code=400, detail="Empty file not allowed")
    
    # Validate file
    try:
        validation_result = validate_file(content, file.filename or "unknown", file.content_type)
        if not validation_result.valid:
            raise HTTPException(
                status_code=415, 
                detail=f"File validation failed: {validation_result.error or 'Invalid file'}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File validation error: {e}")
        raise HTTPException(status_code=422, detail="File validation failed")
    
    # Generate unique attachment ID and file path
    attachment_id = uuid.uuid4().hex
    original_name = file.filename or "unknown"
    mime_type = file.content_type or "application/octet-stream"
    size_bytes = len(content)
    content_hash = _generate_file_hash(content)
    
    # Determine file extension from original filename
    file_ext = Path(original_name).suffix.lower()
    if not file_ext:
        # Fallback extension based on MIME type
        if mime_type.startswith('image/'):
            file_ext = '.jpg'
        elif mime_type == 'application/pdf':
            file_ext = '.pdf'
        else:
            file_ext = '.bin'
    
    # Create session-specific directory and save file
    _ensure_upload_dir()
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    Path(session_dir).mkdir(exist_ok=True)
    
    filename = f"{attachment_id}{file_ext}"
    file_path = os.path.join(session_dir, filename)
    stored_path = os.path.join(session_id, filename)  # Relative path for DB
    
    try:
        with open(file_path, 'wb') as f:
            f.write(content)
    except Exception as e:
        logger.error(f"Failed to save file {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")
    
    # Extract text content after file is saved (extract_text expects a file path)
    extracted_text = None
    try:
        extracted_text = extract_text(file_path, mime_type)
    except Exception as e:
        logger.warning(f"Text extraction failed for {file.filename}: {e}")
        # Continue without extracted text - it's optional
    
    # Save attachment metadata to database
    try:
        attachment_record = database.create_attachment(
            session_id=session_id,
            original_name=original_name,
            stored_path=stored_path,
            mime_type=mime_type,
            size_bytes=size_bytes,
            content_hash=content_hash,
            extracted_text=extracted_text
        )
        
        logger.info(f"File uploaded: {original_name} -> {attachment_id} (user: {user['username']}, session: {session_id})")
        
        return AttachmentResponse(
            id=attachment_record["id"],
            original_name=original_name,
            mime_type=mime_type,
            size_bytes=size_bytes,
            type=_get_file_type(mime_type)
        )
        
    except Exception as e:
        # Clean up file if database save failed
        try:
            os.remove(file_path)
        except OSError:
            pass
        logger.error(f"Database error during file upload: {e}")
        raise HTTPException(status_code=500, detail="Failed to save attachment metadata")


# ── Retrieve Attachment ──────────────────────────────────────────────

@router.get("/v1/attachments/{attachment_id}")
async def get_attachment(
    attachment_id: str,
    user: dict = Depends(get_current_user)
) -> FileResponse:
    """Retrieve an uploaded file.
    
    Args:
        attachment_id: Attachment ID
        user: Current user from JWT token
        
    Returns:
        FileResponse with the file content
        
    Raises:
        HTTPException: 403 (forbidden), 404 (not found)
    """
    user_id = user["user_id"]
    
    # Verify attachment exists and user owns it
    if not _verify_attachment_ownership(attachment_id, user_id):
        raise HTTPException(status_code=404, detail="Attachment not found")
    
    # Get attachment metadata
    attachment = database.get_attachment(attachment_id)
    if not attachment:
        raise HTTPException(status_code=404, detail="Attachment not found")
    
    # Build full file path
    stored_path = attachment["stored_path"]
    full_path = os.path.join(UPLOAD_DIR, stored_path)
    
    # Verify file exists on disk
    if not os.path.exists(full_path):
        logger.error(f"Attachment file not found on disk: {full_path}")
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    # Return file with appropriate headers
    return FileResponse(
        path=full_path,
        media_type=attachment["mime_type"],
        filename=attachment["original_name"],
        headers={
            "Content-Disposition": f'inline; filename="{attachment["original_name"]}"'
        }
    )


# ── Delete Attachment ────────────────────────────────────────────────

@router.delete("/v1/attachments/{attachment_id}", status_code=204)
async def delete_attachment(
    attachment_id: str,
    user: dict = Depends(get_current_user)
) -> None:
    """Delete an uploaded file.
    
    Args:
        attachment_id: Attachment ID
        user: Current user from JWT token
        
    Raises:
        HTTPException: 403 (forbidden), 404 (not found)
    """
    user_id = user["user_id"]
    
    # Verify attachment exists and user owns it
    if not _verify_attachment_ownership(attachment_id, user_id):
        raise HTTPException(status_code=404, detail="Attachment not found")
    
    # Get attachment metadata before deletion
    attachment = database.get_attachment(attachment_id)
    if not attachment:
        raise HTTPException(status_code=404, detail="Attachment not found")
    
    # Delete from database first
    try:
        success = database.delete_attachment(attachment_id)
        if not success:
            raise HTTPException(status_code=404, detail="Attachment not found")
    except Exception as e:
        logger.error(f"Database error during attachment deletion: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete attachment")
    
    # Delete file from disk
    stored_path = attachment["stored_path"]
    full_path = os.path.join(UPLOAD_DIR, stored_path)
    
    try:
        if os.path.exists(full_path):
            os.remove(full_path)
            logger.info(f"Deleted attachment file: {full_path}")
    except OSError as e:
        logger.error(f"Failed to delete file {full_path}: {e}")
        # Don't raise - database deletion succeeded, file cleanup failed
    
    logger.info(f"Attachment deleted: {attachment_id} (user: {user['username']})")


# ── Background Cleanup Task ──────────────────────────────────────────

async def cleanup_orphan_attachments() -> None:
    """Background task to clean up orphaned attachments.
    
    Removes attachments that have no message_id and are older than 1 hour.
    This handles cases where files were uploaded but never attached to messages.
    """
    try:
        # Get orphaned attachments (older than 1 hour)
        orphans = database.get_orphan_attachments(max_age_seconds=3600)
        
        if not orphans:
            return
        
        logger.info(f"Cleaning up {len(orphans)} orphaned attachments")
        
        for attachment in orphans:
            attachment_id = attachment["id"]
            stored_path = attachment["stored_path"]
            full_path = os.path.join(UPLOAD_DIR, stored_path)
            
            # Delete from database
            try:
                database.delete_attachment(attachment_id)
                logger.debug(f"Deleted orphaned attachment from DB: {attachment_id}")
            except Exception as e:
                logger.error(f"Failed to delete orphaned attachment {attachment_id} from DB: {e}")
                continue
            
            # Delete file from disk
            try:
                if os.path.exists(full_path):
                    os.remove(full_path)
                    logger.debug(f"Deleted orphaned file: {full_path}")
            except OSError as e:
                logger.error(f"Failed to delete orphaned file {full_path}: {e}")
        
        logger.info(f"Orphan cleanup completed: {len(orphans)} files processed")
        
    except Exception as e:
        logger.error(f"Error during orphan cleanup: {e}")


def start_cleanup_task() -> None:
    """Start the background cleanup task."""
    async def cleanup_loop():
        """Run cleanup every 30 minutes."""
        while True:
            try:
                await cleanup_orphan_attachments()
                # Wait 30 minutes before next cleanup
                await asyncio.sleep(1800)
            except asyncio.CancelledError:
                logger.info("Cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in cleanup loop: {e}")
                # Wait 5 minutes before retrying
                await asyncio.sleep(300)
    
    # Start the cleanup task
    asyncio.create_task(cleanup_loop())
    logger.info("Started orphan attachment cleanup task")