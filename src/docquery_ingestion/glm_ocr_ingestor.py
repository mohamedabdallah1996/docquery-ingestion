"""GLM-OCR-based ingestion strategy: rasterize each page, send it to the
OCR service via OCRClient, normalize the response into a ParsedPage.
"""

from __future__ import annotations

import asyncio

from docquery_core import ParsedPage

from docquery_ingestion.base import BaseIngestor
from docquery_ingestion.clients.ocr_client import OCRClient
from docquery_ingestion.config import GLMOCRConfig
from docquery_ingestion.utils.rendering import render_page


class GLMOCRIngestor(BaseIngestor):
    def __init__(self, ocr_client: OCRClient, config: GLMOCRConfig) -> None:
        super().__init__(config)
        self._ocr_client = ocr_client
        self._config: GLMOCRConfig = config

    async def _parse_pages(self, pdf_bytes: bytes, page_count: int) -> list[ParsedPage]:
        """Bounded concurrency across pages -- batch_size is OCR-specific,
        not something the common base class needs to know about."""
        semaphore = asyncio.Semaphore(self._config.batch_size)

        async def bounded(page_index: int) -> ParsedPage:
            async with semaphore:
                return await self._parse_page(pdf_bytes, page_index)

        return list(await asyncio.gather(*(bounded(i) for i in range(page_count))))

    async def _parse_page(self, pdf_bytes: bytes, page_index: int) -> ParsedPage:
        image_bytes = render_page(pdf_bytes, page_index, dpi=self._config.render_dpi)
        result = await self._ocr_client.extract_page(image_bytes, page_number=page_index + 1)
        return ParsedPage(page_number=page_index + 1, markdown=result.markdown, error=result.error)
