"""Configuration for the ingestion package.

`IngestionConfig` holds settings common to any ingestion strategy.
`GLMOCRConfig` extends it with settings specific to the GLM-OCR strategy
(model, sampling params, rendering, retry policy, and OCR-call
concurrency) -- these don't belong on the common base, since a future
text-extraction-based strategy wouldn't have an OCR model to configure at
all.

Secrets/endpoints (the OCR service URL) are deliberately not fields here --
they're passed explicitly to the factory, never embedded in this config.
"""

from __future__ import annotations

from pydantic import BaseModel


class IngestionConfig(BaseModel):
    max_file_size_mb: int = 100
    timeout_seconds: float = 300.0  # whole-document processing budget


class GLMOCRConfig(IngestionConfig):
    model: str = "ggml-org/GLM-OCR-GGUF:Q8_0"
    prompt: str = "OCR"
    temperature: float = 0.1
    top_k: int = 1
    render_dpi: int = 200
    request_timeout_s: float = 60.0  # per-page OCR call timeout
    retry_max_attempts: int = 3
    retry_backoff_seconds: float = 2.0
    batch_size: int = 1  # pages OCR'd concurrently within one document
