import pytest

from docquery_ingestion.utils.exceptions import InvalidPDFError
from docquery_ingestion.utils.validation import validate_pdf


def test_accepts_a_real_pdf(one_page_pdf: bytes) -> None:
    validate_pdf(one_page_pdf, max_file_size_mb=100)  # does not raise


def test_rejects_missing_magic_bytes() -> None:
    with pytest.raises(InvalidPDFError, match="magic bytes"):
        validate_pdf(b"this is not a pdf at all", max_file_size_mb=100)


def test_rejects_corrupt_pdf() -> None:
    with pytest.raises(InvalidPDFError, match="could not be opened"):
        validate_pdf(b"%PDF-1.4\ngarbage after a valid header", max_file_size_mb=100)


def test_rejects_oversized_pdf(one_page_pdf: bytes) -> None:
    with pytest.raises(InvalidPDFError, match="exceeds"):
        validate_pdf(one_page_pdf, max_file_size_mb=0)
