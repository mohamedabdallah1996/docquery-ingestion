"""docquery-ingestion: PDF bytes -> ParsedDocument.

Public surface: `build_ingestor()` (the factory), `GLMOCRIngestor` (for
type hints), the config models, and this package's exceptions -- callers
should not need to import anything else here.
"""

from __future__ import annotations

from docquery_ingestion.config import GLMOCRConfig, IngestionConfig
from docquery_ingestion.factory import build_ingestor
from docquery_ingestion.glm_ocr_ingestor import GLMOCRIngestor
from docquery_ingestion.utils.exceptions import InvalidPDFError, OCRServiceError

__all__ = [
    "GLMOCRConfig",
    "GLMOCRIngestor",
    "IngestionConfig",
    "InvalidPDFError",
    "OCRServiceError",
    "build_ingestor",
]
