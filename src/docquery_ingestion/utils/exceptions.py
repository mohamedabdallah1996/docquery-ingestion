"""Exceptions raised by docquery-ingestion, rooted in docquery_core.DocQueryError
so the orchestrator can catch broadly (`except DocQueryError`) without needing
to know about every submodule's specific failure modes.
"""

from __future__ import annotations

from docquery_core import DocQueryError


class InvalidPDFError(DocQueryError):
    """The input isn't a valid, processable PDF (bad magic bytes, corrupt,
    unreadable, or over the configured size limit)."""


class OCRServiceError(DocQueryError):
    """The OCR service failed or timed out after exhausting retries."""
