"""OCR responses -> the canonical ParsedPage/ParsedDocument shape.

Defensive cleanup only (e.g. stripping a code fence a VLM sometimes wraps
its whole answer in) -- this does not parse markdown into structured
blocks, that's chunking's job, not ingestion's.
"""

from __future__ import annotations

import re

from docquery_core import ParsedDocument, ParsedPage, ParseStatus

_CODE_FENCE = re.compile(r"^```(?:markdown)?\n(.*)\n```$", re.DOTALL)


def clean_markdown(text: str) -> str:
    """Strip a wrapping code fence some VLMs add around their whole answer."""
    stripped = text.strip()
    match = _CODE_FENCE.match(stripped)
    return match.group(1).strip() if match else stripped


def assemble_parsed_document(doc_id: str, pages: list[ParsedPage]) -> ParsedDocument:
    """Roll per-page outcomes up into the document's overall status."""
    status: ParseStatus
    if all(page.error for page in pages):
        status = "FAILED"
    elif any(page.error for page in pages):
        status = "PARTIAL"
    else:
        status = "DONE"
    return ParsedDocument(doc_id=doc_id, pages=tuple(pages), status=status)
