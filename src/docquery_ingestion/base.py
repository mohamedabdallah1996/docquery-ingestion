"""Template method base for ingestion strategies.

The shared sequence (validate -> get page count -> parse each page ->
assemble) is common to any strategy; only *how one page's content is
obtained* varies between an OCR-based and a future text-extraction-based
strategy -- that's the one abstract method. `_parse_pages` (the loop) has
a concrete, sequential default that most strategies can use unchanged;
GLMOCRIngestor overrides it to add bounded concurrency (batch_size),
which is an OCR-specific concern, not a common one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from docquery_core import ParsedDocument, ParsedPage
from loguru import logger

from docquery_ingestion.config import IngestionConfig
from docquery_ingestion.utils.normalizer import assemble_parsed_document
from docquery_ingestion.utils.rendering import get_page_count
from docquery_ingestion.utils.validation import validate_pdf


class BaseIngestor(ABC):
    def __init__(self, config: IngestionConfig) -> None:
        self._config = config

    async def ingest(self, pdf_bytes: bytes, doc_id: str) -> ParsedDocument:
        logger.info(
            "ingest started: doc_id={doc_id}, size={size_kb:.1f}KB",
            doc_id=doc_id,
            size_kb=len(pdf_bytes) / 1024,
        )
        validate_pdf(pdf_bytes, max_file_size_mb=self._config.max_file_size_mb)

        page_count = get_page_count(pdf_bytes)
        logger.info("doc_id={doc_id}: {n} page(s) to process", doc_id=doc_id, n=page_count)

        pages = await self._parse_pages(pdf_bytes, page_count)
        document = assemble_parsed_document(doc_id, pages)

        logger.info(
            "ingest finished: doc_id={doc_id}, status={status}, pages={n}",
            doc_id=doc_id,
            status=document.status,
            n=len(pages),
        )
        return document

    async def _parse_pages(self, pdf_bytes: bytes, page_count: int) -> list[ParsedPage]:
        """Default: sequential. Override to add strategy-specific concurrency."""
        return [await self._parse_page(pdf_bytes, i) for i in range(page_count)]

    @abstractmethod
    async def _parse_page(self, pdf_bytes: bytes, page_index: int) -> ParsedPage: ...
