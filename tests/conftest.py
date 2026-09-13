"""Shared fixtures: a real, minimal in-memory PDF -- built with pypdfium2
itself, not a hand-typed byte string, so it's guaranteed to be a PDF this
same library considers valid."""

from __future__ import annotations

import io

import pypdfium2 as pdfium
import pytest


def _build_pdf(page_count: int) -> bytes:
    document = pdfium.PdfDocument.new()
    for _ in range(page_count):
        document.new_page(200, 200)
    buffer = io.BytesIO()
    document.save(buffer)
    document.close()
    return buffer.getvalue()


@pytest.fixture
def one_page_pdf() -> bytes:
    return _build_pdf(1)


@pytest.fixture
def three_page_pdf() -> bytes:
    return _build_pdf(3)
