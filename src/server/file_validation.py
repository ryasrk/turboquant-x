"""File validation module for TurboQuant-X attachment uploads.

Validates uploaded files before storage using header-based MIME detection,
size limits, filename sanitization, image verification, and content hashing.
All uploaded content is treated as untrusted.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import python-magic for header-based MIME detection.
# Fall back to extension-based guessing if unavailable.
# ---------------------------------------------------------------------------
try:
    import magic

    _HAS_MAGIC = True
except ImportError:
    _HAS_MAGIC = False
    logger.warning(
        "python-magic is not installed — falling back to extension-based MIME "
        "detection. Install python-magic (plus libmagic) for header-based "
        "validation: pip install python-magic"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FILE_SIZE: int = 20 * 1024 * 1024  # 20 MB

ALLOWED_MIMES: dict[str, list[str]] = {
    # Images
    "image/png": [".png"],
    "image/jpeg": [".jpg", ".jpeg"],
    "image/gif": [".gif"],
    "image/webp": [".webp"],
    # Documents
    "application/pdf": [".pdf"],
    "text/plain": [".txt", ".text"],
    "text/markdown": [".md", ".markdown"],
    "text/csv": [".csv"],
    "application/json": [".json"],
    "text/x-python": [".py"],
    "text/x-yaml": [".yaml", ".yml"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
        ".docx"
    ],
}

_IMAGE_MIMES: frozenset[str] = frozenset(
    m for m in ALLOWED_MIMES if m.startswith("image/")
)

# Reverse lookup: extension -> MIME
_EXT_TO_MIME: dict[str, str] = {}
for _mime, _exts in ALLOWED_MIMES.items():
    for _ext in _exts:
        _EXT_TO_MIME[_ext] = _mime

# Filename sanitisation pattern — allow only safe characters.
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9._\-]")

# Maximum length for sanitised filenames.
_MAX_FILENAME_LENGTH = 255


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Outcome of file validation."""

    valid: bool
    sanitized_name: str
    detected_mime: str
    content_hash: str
    size_bytes: int
    file_type: str  # 'image' or 'document'
    error: str | None = None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def sanitize_filename(name: str) -> str:
    """Sanitise a user-supplied filename for safe storage.

    * Strips directory / path components (prevents path-traversal attacks).
    * Replaces spaces with underscores.
    * Removes any character that is not alphanumeric, dot, underscore or hyphen.
    * Truncates to 255 characters.
    * Ensures the result is never empty.
    """
    # Take only the basename — discard any directory components.
    # os.path is intentionally avoided so embedded NUL bytes / OS tricks
    # cannot influence splitting; we split on the common separators.
    for sep in ("/", "\\"):
        name = name.rsplit(sep, maxsplit=1)[-1]

    # Replace spaces before stripping other chars.
    name = name.replace(" ", "_")

    # Remove everything outside the safe set.
    name = _SAFE_FILENAME_RE.sub("", name)

    # Strip leading dots to prevent hidden-file creation.
    name = name.lstrip(".")

    # Enforce length limit.
    if len(name) > _MAX_FILENAME_LENGTH:
        # Preserve the extension when truncating.
        dot_idx = name.rfind(".")
        if dot_idx > 0:
            ext = name[dot_idx:]
            stem_limit = _MAX_FILENAME_LENGTH - len(ext)
            if stem_limit > 0:
                name = name[:stem_limit] + ext
            else:
                name = name[:_MAX_FILENAME_LENGTH]
        else:
            name = name[:_MAX_FILENAME_LENGTH]

    if not name:
        name = "unnamed_upload"

    return name


def compute_hash(content: bytes) -> str:
    """Return the SHA-256 hex digest of *content*."""
    return hashlib.sha256(content).hexdigest()


def classify_file_type(mime: str) -> str:
    """Classify a MIME type as ``'image'`` or ``'document'``."""
    return "image" if mime in _IMAGE_MIMES else "document"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_mime_magic(content: bytes) -> str | None:
    """Detect MIME via libmagic header inspection."""
    if not _HAS_MAGIC:
        return None
    try:
        detected = magic.from_buffer(content, mime=True)
        return detected if isinstance(detected, str) else None
    except Exception:
        logger.exception("python-magic MIME detection failed")
        return None


def _detect_mime_by_extension(filename: str) -> str | None:
    """Guess MIME from the file extension (fallback only)."""
    dot_idx = filename.rfind(".")
    if dot_idx < 0:
        return None
    ext = filename[dot_idx:].lower()
    return _EXT_TO_MIME.get(ext)


def _extension_of(filename: str) -> str:
    """Return the lowercased extension including the dot, or ``''``."""
    dot_idx = filename.rfind(".")
    return filename[dot_idx:].lower() if dot_idx >= 0 else ""


def _verify_image(content: bytes, mime: str) -> str | None:
    """Open the image with Pillow to confirm it decodes.

    Returns an error string on failure, or ``None`` on success.
    Prevents polyglot attacks (e.g. JPEG header + embedded script).
    """
    try:
        from PIL import Image

        # Guard against decompression bombs.
        Image.MAX_IMAGE_PIXELS = 25_000_000

        img = Image.open(__import__("io").BytesIO(content))
        try:
            # Force full decode — .verify() reads headers only on some
            # formats, so we load pixels to be sure.
            img.load()
        finally:
            img.close()
    except ImportError:
        logger.warning("Pillow is not installed — skipping image verification")
        return None
    except Exception as exc:
        return f"Image failed to decode ({mime}): {exc}"

    return None


# ---------------------------------------------------------------------------
# Main validation entry-point
# ---------------------------------------------------------------------------


def validate_file(
    file_content: bytes,
    original_filename: str,
    claimed_mime: str | None = None,
) -> ValidationResult:
    """Validate an uploaded file and return a :class:`ValidationResult`.

    Parameters
    ----------
    file_content:
        Raw bytes of the uploaded file.
    original_filename:
        The filename provided by the client (untrusted).
    claimed_mime:
        Optional browser-provided MIME type. Used only as a hint — the
        authoritative type is determined by libmagic header inspection.

    Returns
    -------
    ValidationResult
        ``valid=True`` when the file passes all checks, otherwise
        ``valid=False`` with a human-readable ``error`` message.
    """
    size = len(file_content)
    safe_name = sanitize_filename(original_filename)
    content_hash = compute_hash(file_content)

    # Convenience for early-exit error results.
    def _fail(msg: str) -> ValidationResult:
        return ValidationResult(
            valid=False,
            sanitized_name=safe_name,
            detected_mime="",
            content_hash=content_hash,
            size_bytes=size,
            file_type="",
            error=msg,
        )

    # ------------------------------------------------------------------
    # 1. Size check
    # ------------------------------------------------------------------
    if size == 0:
        return _fail("File is empty.")
    if size > MAX_FILE_SIZE:
        limit_mb = MAX_FILE_SIZE / (1024 * 1024)
        return _fail(
            f"File exceeds the {limit_mb:.0f} MB size limit "
            f"({size / (1024 * 1024):.1f} MB uploaded)."
        )

    # ------------------------------------------------------------------
    # 2. MIME detection (header-based preferred, extension fallback)
    # ------------------------------------------------------------------
    detected_mime = _detect_mime_magic(file_content)
    used_fallback = False

    if detected_mime is None:
        # Fallback to extension-based detection.
        detected_mime = _detect_mime_by_extension(safe_name)
        used_fallback = True
        if detected_mime is None:
            return _fail(
                "Unable to determine file type. "
                "Ensure python-magic is installed or provide a known extension."
            )

    # ------------------------------------------------------------------
    # 3. Allowlist check
    # ------------------------------------------------------------------
    if detected_mime not in ALLOWED_MIMES:
        return _fail(
            f"File type '{detected_mime}' is not allowed. "
            f"Accepted types: {', '.join(sorted(ALLOWED_MIMES))}."
        )

    # ------------------------------------------------------------------
    # 4. Extension ↔ MIME consistency
    # ------------------------------------------------------------------
    ext = _extension_of(safe_name)
    allowed_exts = ALLOWED_MIMES[detected_mime]

    if ext and ext not in allowed_exts:
        return _fail(
            f"Extension '{ext}' does not match detected type "
            f"'{detected_mime}' (expected {', '.join(allowed_exts)})."
        )

    # ------------------------------------------------------------------
    # 5. Image verification (Pillow decode)
    # ------------------------------------------------------------------
    file_type = classify_file_type(detected_mime)

    if file_type == "image":
        img_err = _verify_image(file_content, detected_mime)
        if img_err is not None:
            return _fail(img_err)

    # ------------------------------------------------------------------
    # 6. Claimed-MIME cross-check (advisory — log only when fallback used)
    # ------------------------------------------------------------------
    if (
        claimed_mime is not None
        and claimed_mime != detected_mime
        and not used_fallback
    ):
        logger.info(
            "Claimed MIME '%s' differs from detected '%s' for file '%s'",
            claimed_mime,
            detected_mime,
            safe_name,
        )

    return ValidationResult(
        valid=True,
        sanitized_name=safe_name,
        detected_mime=detected_mime,
        content_hash=content_hash,
        size_bytes=size,
        file_type=file_type,
    )
