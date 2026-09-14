"""OCR client for llama.cpp's OpenAI-compatible GLM-OCR endpoint.

This is a thin HTTP client; there is no custom OCR server here. Requests are
retried at the page level, not the whole document. See
``docquery_ingestion.base.BaseIngestor``.
"""

from __future__ import annotations

import base64

import httpx
from loguru import logger
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from docquery_ingestion.clients.ocr_client import PageOCRResult
from docquery_ingestion.utils.normalizer import clean_markdown

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.TransportError,
)

_PAGE_FAILURE = (
    *_RETRYABLE_EXCEPTIONS,
    httpx.HTTPStatusError,
    IndexError,
    KeyError,
    ValueError,
    TypeError,
)


def _is_retryable_exception(exception: BaseException) -> bool:
    """Return whether an exception represents a transient OCR failure."""
    if isinstance(exception, _RETRYABLE_EXCEPTIONS):
        return True

    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code in _RETRYABLE_STATUS_CODES

    return False


class LlamaCppOCRClient:
    """HTTP client for a llama.cpp OpenAI-compatible GLM-OCR endpoint.

    One HTTP client and connection pool is maintained for the lifetime of
    this instance. The client should therefore normally be created once and
    reused across documents.
    """

    def __init__(
        self,
        service_url: str,
        *,
        model: str = "ggml-org/GLM-OCR-GGUF:Q8_0",
        prompt: str = "OCR",
        temperature: float = 0.1,
        top_k: int = 1,
        request_timeout_s: float = 60.0,
        retry_max_attempts: int = 3,
        retry_backoff_seconds: float = 2.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._service_url = service_url.rstrip("/")
        self._model = model
        self._prompt = prompt
        self._temperature = temperature
        self._top_k = top_k
        self._retrying = AsyncRetrying(
            retry=retry_if_exception(_is_retryable_exception),
            stop=stop_after_attempt(retry_max_attempts),
            wait=wait_exponential(multiplier=retry_backoff_seconds),
            before_sleep=self._log_retry,
            reraise=True,
        )
        self._http_client = httpx.AsyncClient(
            timeout=request_timeout_s,
            transport=transport,
        )

    async def __aenter__(self) -> LlamaCppOCRClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP client and its connection pool."""
        await self._http_client.aclose()

    @staticmethod
    def _log_retry(retry_state: RetryCallState) -> None:
        outcome = retry_state.outcome
        exception = outcome.exception() if outcome is not None else None

        logger.warning(
            "OCR request failed (attempt {attempt}), retrying: {exception}",
            attempt=retry_state.attempt_number,
            exception=exception,
        )

    async def extract_page(
        self,
        image_bytes: bytes,
        *,
        page_number: int,
    ) -> PageOCRResult:
        """Run OCR on one page and return a graceful failure on page errors."""
        try:
            content: str = await self._retrying(
                self._request_ocr,
                image_bytes,
            )
        except _PAGE_FAILURE as exc:
            logger.error(
                "OCR failed for page {page}: {exception}",
                page=page_number,
                exception=exc,
            )
            return PageOCRResult(
                page_number=page_number,
                markdown="",
                error=str(exc),
            )

        return PageOCRResult(
            page_number=page_number,
            markdown=clean_markdown(content),
        )

    async def _request_ocr(self, image_bytes: bytes) -> str:
        """Send one OCR request and return the raw markdown response."""
        payload = self._build_payload(image_bytes)

        response = await self._http_client.post(
            f"{self._service_url}/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()

        return self._parse_response(response)

    def _build_payload(self, image_bytes: bytes) -> dict[str, object]:
        """Build the OpenAI-compatible multimodal OCR request payload."""
        encoded_image = base64.b64encode(image_bytes).decode("ascii")

        return {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}",
                            },
                        },
                        {
                            "type": "text",
                            "text": self._prompt,
                        },
                    ],
                }
            ],
            "temperature": self._temperature,
            "top_k": self._top_k,
        }

    @staticmethod
    def _parse_response(response: httpx.Response) -> str:
        """Extract OCR markdown from the llama.cpp response.

        Raises TypeError if `content` isn't a string -- e.g. a valid-shaped
        response with `content: null` -- rather than silently coercing it
        into the literal text "None" via str(...).
        """
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError(f"expected a string 'content', got {type(content).__name__}")
        return content
