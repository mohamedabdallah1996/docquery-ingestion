from docquery_ingestion.clients.base_client import PageOCRResult
from docquery_ingestion.config import GLMOCRConfig
from docquery_ingestion.ingestor import GLMOCRIngestor
from docquery_ingestion.utils.exceptions import InvalidPDFError


class FakeOCRClient:
    """A fake, not a mock -- implements OCRClient structurally, no framework needed."""

    def __init__(self, fail_pages: set[int] | None = None) -> None:
        self._fail_pages = fail_pages or set()
        self.calls: list[int] = []

    async def extract_page(self, image_bytes: bytes, *, page_number: int) -> PageOCRResult:
        self.calls.append(page_number)
        if page_number in self._fail_pages:
            return PageOCRResult(page_number=page_number, markdown="", error="simulated failure")
        return PageOCRResult(page_number=page_number, markdown=f"# Page {page_number}")


def _ingestor(fake_client: FakeOCRClient, **overrides: object) -> GLMOCRIngestor:
    config = GLMOCRConfig(**overrides)
    return GLMOCRIngestor(ocr_client=fake_client, config=config)


async def test_ingest_a_valid_pdf_returns_done(three_page_pdf: bytes) -> None:
    fake_client = FakeOCRClient()
    ingestor = _ingestor(fake_client)

    document = await ingestor.ingest(three_page_pdf, doc_id="doc-1")

    assert document.doc_id == "doc-1"
    assert document.status == "DONE"
    assert [p.page_number for p in document.pages] == [1, 2, 3]
    assert document.pages[0].markdown == "# Page 1"
    assert sorted(fake_client.calls) == [1, 2, 3]


async def test_ingest_with_one_failed_page_is_partial(three_page_pdf: bytes) -> None:
    fake_client = FakeOCRClient(fail_pages={2})
    ingestor = _ingestor(fake_client)

    document = await ingestor.ingest(three_page_pdf, doc_id="doc-1")

    assert document.status == "PARTIAL"
    failed_page = next(p for p in document.pages if p.page_number == 2)
    assert failed_page.error == "simulated failure"


async def test_ingest_rejects_an_invalid_pdf_before_calling_ocr() -> None:
    fake_client = FakeOCRClient()
    ingestor = _ingestor(fake_client)

    try:
        await ingestor.ingest(b"not a pdf", doc_id="doc-1")
        raise AssertionError("expected InvalidPDFError")
    except InvalidPDFError:
        pass

    assert fake_client.calls == []  # never got as far as calling OCR


async def test_batch_size_bounds_concurrency(three_page_pdf: bytes) -> None:
    """Not asserting exact scheduling order -- just that a batch_size=1
    ingestor still processes every page, sequentially, with no crashes."""
    fake_client = FakeOCRClient()
    ingestor = _ingestor(fake_client, batch_size=1)

    document = await ingestor.ingest(three_page_pdf, doc_id="doc-1")

    assert document.status == "DONE"
    assert len(document.pages) == 3
