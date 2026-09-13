"""PDF -> page image bytes. Only used by OCR-based ingestion strategies --
a future text-extraction-based strategy wouldn't rasterize anything."""

from __future__ import annotations

import io

import pypdfium2 as pdfium

_POINTS_PER_INCH = 72


def get_page_count(pdf_bytes: bytes) -> int:
    document = pdfium.PdfDocument(pdf_bytes)
    try:
        return len(document)
    finally:
        document.close()


def render_page(pdf_bytes: bytes, page_index: int, *, dpi: int) -> bytes:
    """Render one page (0-indexed) to JPEG bytes at the given DPI."""
    scale = dpi / _POINTS_PER_INCH
    document = pdfium.PdfDocument(pdf_bytes)
    try:
        page = document[page_index]
        try:
            bitmap = page.render(scale=scale)
            try:
                pil_image = bitmap.to_pil().convert("RGB")
            finally:
                bitmap.close()
        finally:
            page.close()
    finally:
        document.close()

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()
