from docquery_ingestion.utils.rendering import get_page_count, render_page

_JPEG_MAGIC = b"\xff\xd8"


def test_get_page_count(three_page_pdf: bytes) -> None:
    assert get_page_count(three_page_pdf) == 3


def test_render_page_produces_a_jpeg(one_page_pdf: bytes) -> None:
    image_bytes = render_page(one_page_pdf, 0, dpi=100)
    assert image_bytes.startswith(_JPEG_MAGIC)


def test_higher_dpi_produces_more_bytes(one_page_pdf: bytes) -> None:
    low = render_page(one_page_pdf, 0, dpi=72)
    high = render_page(one_page_pdf, 0, dpi=300)
    assert len(high) > len(low)
