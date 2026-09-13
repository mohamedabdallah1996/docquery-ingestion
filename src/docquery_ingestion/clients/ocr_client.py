"""Boundary to the OCR backend -- currently llama.cpp serving GLM-OCR
(docquery/services/ocr). This Protocol is what lets the backend be swapped
later (RunPod, Triton, ...) without touching the ingestor that calls it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class PageOCRResult(BaseModel):
    """One page's OCR outcome. `error` set (and `markdown` empty) means the
    page failed after exhausting retries -- the caller decides how a
    per-page failure affects the overall document (see normalizer.py)."""

    page_number: int
    markdown: str
    error: str | None = None


@runtime_checkable
class OCRClient(Protocol):
    async def extract_page(self, image_bytes: bytes, *, page_number: int) -> PageOCRResult: ...
