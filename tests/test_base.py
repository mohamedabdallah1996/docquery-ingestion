import pytest
from docquery_core import ParsedPage

from docquery_ingestion.base import BaseIngestor
from docquery_ingestion.config import IngestionConfig


class MinimalIngestor(BaseIngestor):
    """The simplest possible strategy: every page becomes trivial markdown.
    Exercises the template method's shared sequence without any OCR."""

    async def _parse_page(self, pdf_bytes: bytes, page_index: int) -> ParsedPage:
        return ParsedPage(page_number=page_index + 1, markdown=f"page {page_index + 1}")


def test_base_ingestor_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        BaseIngestor(IngestionConfig())  # type: ignore[abstract]


async def test_template_method_validates_counts_pages_and_assembles(three_page_pdf: bytes) -> None:
    ingestor = MinimalIngestor(IngestionConfig())

    document = await ingestor.ingest(three_page_pdf, doc_id="doc-1")

    assert document.doc_id == "doc-1"
    assert document.status == "DONE"
    assert [p.markdown for p in document.pages] == ["page 1", "page 2", "page 3"]
