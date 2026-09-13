"""PDF structural validation -- always run, no toggle (per design decision:
skipping validation is a footgun, not a legitimate configuration)."""

from __future__ import annotations

import pypdfium2 as pdfium
from loguru import logger

from docquery_ingestion.utils.exceptions import InvalidPDFError

PDF_MAGIC_BYTES = b"%PDF-"


def validate_pdf(pdf_bytes: bytes, *, max_file_size_mb: int) -> None:
    """Raise InvalidPDFError if `pdf_bytes` isn't a processable PDF."""
    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > max_file_size_mb:
        logger.warning(
            "rejected PDF: {size:.1f}MB exceeds the {limit}MB limit",
            size=size_mb,
            limit=max_file_size_mb,
        )
        raise InvalidPDFError(f"PDF is {size_mb:.1f}MB, exceeds the {max_file_size_mb}MB limit")

    if not pdf_bytes.startswith(PDF_MAGIC_BYTES):
        logger.warning("rejected input: missing PDF magic bytes")
        raise InvalidPDFError("input does not start with the PDF magic bytes")

    try:
        document = pdfium.PdfDocument(pdf_bytes)
    except pdfium.PdfiumError as exc:
        logger.warning("rejected PDF: failed to open ({exc})", exc=exc)
        raise InvalidPDFError(f"PDF could not be opened: {exc}") from exc

    try:
        page_count = len(document)
    finally:
        document.close()

    if page_count == 0:
        logger.warning("rejected PDF: zero pages")
        raise InvalidPDFError("PDF has no pages")
