"""GLM-OCR-based ingestion: validate a PDF, rasterize each page, send it to
the OCR service via OCRClient, and assemble the normalized result.

One concrete class, no strategy hierarchy -- v1 has exactly one ingestion
approach (GLM-OCR). If a second one (e.g. text-extraction for digitally
native PDFs) is ever actually built, factor the shared sequence out into a
base class *then*, when there's a second real implementation to justify it.

The OCR backend itself stays swappable independently of that: OCRClient is
still a Protocol, injected here, not something this class owns -- see
clients/ocr_client.py for why that boundary is justified regardless of how
many ingestion strategies exist.
"""

from __future__ import annotations

import asyncio

from docquery_core import ParsedDocument, ParsedPage
from loguru import logger

from docquery_ingestion.clients.ocr_client import OCRClient
from docquery_ingestion.config import GLMOCRConfig
from docquery_ingestion.utils.normalizer import assemble_parsed_document
from docquery_ingestion.utils.rendering import get_page_count, render_page
from docquery_ingestion.utils.validation import validate_pdf


class GLMOCRIngestor:
    def __init__(self, ocr_client: OCRClient, config: GLMOCRConfig) -> None:
        self._ocr_client = ocr_client
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
        """Bounded concurrency across pages (batch_size)."""
        semaphore = asyncio.Semaphore(self._config.batch_size)

        async def bounded(page_index: int) -> ParsedPage:
            async with semaphore:
                return await self._parse_page(pdf_bytes, page_index)

        return list(await asyncio.gather(*(bounded(i) for i in range(page_count))))

    async def _parse_page(self, pdf_bytes: bytes, page_index: int) -> ParsedPage:
        image_bytes = render_page(pdf_bytes, page_index, dpi=self._config.render_dpi)
        result = await self._ocr_client.extract_page(image_bytes, page_number=page_index + 1)
        return ParsedPage(page_number=page_index + 1, markdown=result.markdown, error=result.error)
