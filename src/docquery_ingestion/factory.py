"""Resolves the full ingestion module config into a ready-to-use ingestor.

This is the one place "which strategy/backend" gets decided -- callers
should never construct GLMOCRIngestor or LlamaCppOCRClient directly.
Only one strategy exists today (GLM-OCR); this stays a single-branch
factory until a second one actually exists to select between.

Secrets/endpoints (the OCR service URL) are passed explicitly, never
embedded in `config` -- see config.py.
"""

from __future__ import annotations

from typing import Any

from docquery_ingestion.clients.glm_ocr import LlamaCppOCRClient
from docquery_ingestion.config import GLMOCRConfig
from docquery_ingestion.ingestor import GLMOCRIngestor


def build_ingestor(config: dict[str, Any], *, ocr_service_url: str) -> GLMOCRIngestor:
    """Build the configured ingestor from the orchestrator's full
    `{"ingestion": {...}}`-shaped config."""
    parsed = GLMOCRConfig.model_validate(config["ingestion"])
    ocr_client = LlamaCppOCRClient(
        service_url=ocr_service_url,
        model=parsed.model,
        prompt=parsed.prompt,
        temperature=parsed.temperature,
        top_k=parsed.top_k,
        request_timeout_s=parsed.request_timeout_s,
        retry_max_attempts=parsed.retry_max_attempts,
        retry_backoff_seconds=parsed.retry_backoff_seconds,
    )
    return GLMOCRIngestor(ocr_client=ocr_client, config=parsed)
