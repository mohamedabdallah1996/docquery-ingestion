# docquery-ingestion

> PDF bytes -> `ParsedDocument`. One of DocQuery's independently versioned pipeline submodules.

## Overview

Takes raw PDF bytes and a document ID, validates and rasterizes the PDF, sends each page
through GLM-OCR (served locally by [`llama.cpp`](https://github.com/ggml-org/llama.cpp) --
see [Running the OCR service](#running-the-ocr-service) below), and returns a
[`docquery_core.ParsedDocument`](https://github.com/mohamedabdallah1996/docquery-core) --
the shared type that downstream chunking, embedding, and generation submodules consume. No
submodule-to-submodule calls: this package takes input, produces output, nothing more.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12+ |
| PDF handling | pypdfium2 |
| HTTP | httpx |
| Retries | tenacity |
| Logging | loguru |
| Data contracts | docquery-core (pydantic) |
| Package/dependency management | uv |
| Linting & formatting | Ruff |
| Type checking | mypy (`--strict`) |
| Testing | pytest + pytest-asyncio |

## Architecture

A Template Method hierarchy, not a flat function: `BaseIngestor` owns the shared sequence
(validate -> get page count -> parse each page -> assemble the result) as concrete methods;
only *how one page's content is obtained* is abstract (`_parse_page`). This is deliberate --
a future text-extraction-based strategy (for digitally-native PDFs that don't need OCR at
all) would subclass `BaseIngestor` and implement `_parse_page` differently, reusing
everything else unchanged.

`GLMOCRIngestor` is the only concrete strategy today. It also overrides `_parse_pages` (the
loop, not just one page) to add bounded concurrency across pages within one document
(`batch_size`) -- concurrency control is OCR-specific, not something a future non-OCR
strategy would need, so it isn't on the common base.

The OCR backend itself is a second, independent axis of variation, kept separate from the
strategy hierarchy above: `OCRClient` is a `Protocol` (`clients/ocr_client.py`);
`LlamaCppOCRClient` (`clients/glm_ocr.py`) is its only implementation, calling
`llama.cpp`'s OpenAI-compatible endpoint. Swapping the backend later (RunPod, Triton) means
adding a new `OCRClient` implementation, not touching `GLMOCRIngestor`.

Retries are page-level, inside `LlamaCppOCRClient`, not document-level in some outer layer --
if page 7 of 10 fails transiently, only page 7 retries; pages 1-6 aren't redone. A page that
still fails after retries (or gets a malformed response) degrades gracefully: it's recorded
with an `error`, and `ingest()` returns a normal `ParsedDocument` with `status="PARTIAL"` or
`"FAILED"` -- it never raises for an OCR problem. `InvalidPDFError` is the exception that
does raise, for a structurally bad input, checked once up front before any OCR call happens.

`build_ingestor()` (`factory.py`) is the one place "which strategy" gets decided from a raw
config dict -- callers never construct `GLMOCRIngestor` or `LlamaCppOCRClient` directly. It
stays a single-branch factory until a second real strategy exists to select between.

## Project Structure

```
src/docquery_ingestion/
  ingestor.py            # GLMOCRIngestor -- one concrete ingestor, no strategy hierarchy
  config.py                # IngestionConfig (common) / GLMOCRConfig (OCR-specific)
  factory.py                 # build_ingestor() -- resolves config -> a ready ingestor
  clients/
    base_client.py               # OCRBaseClient Protocol + PageOCRResult
    glm_ocr.py                     # LlamaCppOCRClient -- the only OCRBaseClient today
  utils/
    validation.py                    # PDF structural validation, always run
    rendering.py                       # PDF page -> JPEG bytes (pypdfium2)
    normalizer.py                        # OCR response -> ParsedPage/ParsedDocument
    exceptions.py                          # InvalidPDFError, OCRServiceError
tests/                                       # one test module per src file
Dockerfile               # the OCR service -- official llama.cpp image, no custom serving code
docker-compose.yml          # `docker compose up` runs it, with the GPU/volume/healthcheck config
```

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

### Installation

```bash
git clone https://github.com/mohamedabdallah1996/docquery-ingestion
cd docquery-ingestion
uv sync
```

### Using it

```python
from docquery_ingestion import build_ingestor

ingestor = build_ingestor(
    {"ingestion": {"max_file_size_mb": 100, "batch_size": 4}},
    ocr_service_url="http://localhost:8080",
)
document = await ingestor.ingest(pdf_bytes, doc_id="...")
```

`docquery-core` is depended on via a git source (`[tool.uv.sources]`), not a local path --
this package's own checkout has no sibling `docquery-core` directory to point a path at.
The orchestrator overrides this the same way for its own reasons; see
[`docquery`'s README](https://github.com/mohamedabdallah1996/DocQuery) for that side of it.

### Running the OCR service

This repo is self-contained: `Dockerfile` + `docker-compose.yml` run GLM-OCR locally via the
official [`llama.cpp`](https://github.com/ggml-org/llama.cpp) CUDA server image -- no custom
serving code, nothing to install by hand. Requires an NVIDIA GPU and the NVIDIA container
toolkit. Just one service, so no need to name it:

```bash
docker compose up
```

First run downloads the model (a few hundred MB); subsequent runs reuse it via the
`glm-ocr-cache` volume. Once it's up, `LlamaCppOCRClient(...).verify()` confirms it's actually
ready (checks llama.cpp's own `/health`, which reflects GPU/model-load failures on the server
side) before you send it real requests -- called explicitly, not automatically at construction.

Manual single-page smoke test against a real PDF, once the service is running:

```bash
uv run python -m docquery_ingestion.clients.glm_ocr http://localhost:8080 /path/to/file.pdf
```

## Testing

```bash
uv run pytest
```

Every OCR-facing test uses `httpx.MockTransport` (dependency-injected via `LlamaCppOCRClient`'s
`transport` parameter) or a hand-written `FakeOCRClient` implementing the `OCRClient` Protocol
structurally -- no real GPU or running OCR service needed. Covers: PDF validation
(accept/reject-garbage/reject-corrupt/reject-oversized), rendering, response normalization
(code-fence stripping, status roll-up), the OCR client's retry/graceful-degradation behavior,
and the full `ingest()` sequence end to end via a fake client.

## Development

```bash
uv run ruff check .
uv run ruff format .
uv run mypy src
```
