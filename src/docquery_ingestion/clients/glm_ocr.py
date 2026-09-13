"""OCRClient implementation calling llama.cpp's OpenAI-compatible endpoint
serving GLM-OCR (docquery/services/ocr). No custom OCR server here -- this
is a thin HTTP client; retries happen at the page level, not the whole
document (see docquery_ingestion.base.BaseIngestor).
"""

from __future__ import annotations

import base64

import httpx
from loguru import logger
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from docquery_ingestion.clients.ocr_client import PageOCRResult
from docquery_ingestion.utils.normalizer import clean_markdown

# Transient -- tenacity retries these, at the page level, up to retry_max_attempts.
_RETRYABLE = (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError)
# Anything else that can go wrong extracting a page (a malformed response
# body, a missing key) is NOT retried -- it won't succeed on retry -- but
# still degrades that one page gracefully rather than crashing the whole
# document's ingest().
_PAGE_FAILURE = (*_RETRYABLE, KeyError, ValueError, TypeError)


class LlamaCppOCRClient:
    def __init__(
        self,
        service_url: str,
        model: str,
        prompt: str,
        temperature: float,
        top_k: int,
        request_timeout_s: float,
        retry_max_attempts: int,
        retry_backoff_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._service_url = service_url.rstrip("/")
        self._model = model
        self._prompt = prompt
        self._temperature = temperature
        self._top_k = top_k
        self._timeout = request_timeout_s
        self._transport = transport  # injectable for tests; None -> httpx's real transport
        self._retrying = AsyncRetrying(
            retry=retry_if_exception_type(_RETRYABLE),
            stop=stop_after_attempt(retry_max_attempts),
            wait=wait_exponential(multiplier=retry_backoff_seconds),
            before_sleep=self._log_retry,
            reraise=True,
        )

    @staticmethod
    def _log_retry(retry_state: object) -> None:
        attempt = getattr(retry_state, "attempt_number", "?")
        outcome = getattr(retry_state, "outcome", None)
        exc = outcome.exception() if outcome is not None else None
        logger.warning(
            "OCR request failed (attempt {attempt}), retrying: {exc}", attempt=attempt, exc=exc
        )

    async def extract_page(self, image_bytes: bytes, *, page_number: int) -> PageOCRResult:
        try:
            content: str = await self._retrying(self._request, image_bytes)
        except _PAGE_FAILURE as exc:
            logger.error("OCR failed for page {page}: {exc}", page=page_number, exc=exc)
            return PageOCRResult(page_number=page_number, markdown="", error=str(exc))
        return PageOCRResult(page_number=page_number, markdown=clean_markdown(content))

    async def _request(self, image_bytes: bytes) -> str:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": self._prompt},
                    ],
                }
            ],
            "temperature": self._temperature,
            "top_k": self._top_k,
        }
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            response = await client.post(f"{self._service_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return str(data["choices"][0]["message"]["content"])
