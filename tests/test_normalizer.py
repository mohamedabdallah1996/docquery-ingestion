from docquery_core import ParsedPage

from docquery_ingestion.utils.normalizer import assemble_parsed_document, clean_markdown


def test_clean_markdown_strips_a_markdown_code_fence() -> None:
    assert clean_markdown("```markdown\n# Title\ncontent\n```") == "# Title\ncontent"


def test_clean_markdown_strips_a_bare_code_fence() -> None:
    assert clean_markdown("```\n# Title\n```") == "# Title"


def test_clean_markdown_leaves_unfenced_text_untouched() -> None:
    assert clean_markdown("# Title\ncontent") == "# Title\ncontent"


def test_assemble_all_pages_succeeded_is_done() -> None:
    pages = [ParsedPage(page_number=1, markdown="a"), ParsedPage(page_number=2, markdown="b")]
    document = assemble_parsed_document("doc-1", pages)
    assert document.status == "DONE"


def test_assemble_some_pages_failed_is_partial() -> None:
    pages = [
        ParsedPage(page_number=1, markdown="a"),
        ParsedPage(page_number=2, markdown="", error="timed out"),
    ]
    document = assemble_parsed_document("doc-1", pages)
    assert document.status == "PARTIAL"


def test_assemble_all_pages_failed_is_failed() -> None:
    pages = [ParsedPage(page_number=1, markdown="", error="down")]
    document = assemble_parsed_document("doc-1", pages)
    assert document.status == "FAILED"
