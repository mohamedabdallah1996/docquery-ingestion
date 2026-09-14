import httpx
import pytest

from docquery_ingestion.clients.glm_ocr import LlamaCppOCRClient
from docquery_ingestion.utils.exceptions import OCRServiceError


def _client(handler, *, retry_max_attempts: int = 3) -> LlamaCppOCRClient:
    return LlamaCppOCRClient(
        service_url="http://fake",
        model="m",
        prompt="OCR",
        temperature=0.1,
        top_k=1,
        request_timeout_s=5,
        retry_max_attempts=retry_max_attempts,
        retry_backoff_seconds=0.01,  # fast retries in tests
        transport=httpx.MockTransport(handler),
    )


async def test_successful_call_returns_cleaned_markdown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "```markdown\n# Hi\n```"}}]}
        )

    client = _client(handler)
    result = await client.extract_page(b"image", page_number=1)

    assert result.error is None
    assert result.markdown == "# Hi"
    assert result.page_number == 1


async def test_transient_failure_recovers_on_retry() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = _client(handler, retry_max_attempts=5)
    result = await client.extract_page(b"image", page_number=1)

    assert calls["n"] == 3
    assert result.error is None
    assert result.markdown == "ok"


async def test_exhausted_retries_degrade_gracefully_not_an_exception() -> None:
    def always_503(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = _client(always_503, retry_max_attempts=2)
    result = await client.extract_page(b"image", page_number=7)  # must not raise

    assert result.page_number == 7
    assert result.markdown == ""
    assert result.error is not None


async def test_malformed_response_degrades_gracefully_without_retrying() -> None:
    calls = {"n": 0}

    def malformed(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"unexpected": "shape"})

    client = _client(malformed, retry_max_attempts=5)
    result = await client.extract_page(b"image", page_number=1)

    assert calls["n"] == 1  # not retried -- a malformed body won't fix itself
    assert result.error is not None


async def test_empty_choices_list_degrades_gracefully() -> None:
    """A plausible malformed response (valid JSON, empty `choices`) must not
    crash the whole document via an uncaught IndexError."""

    def empty_choices(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = _client(empty_choices)
    result = await client.extract_page(b"image", page_number=1)  # must not raise

    assert result.markdown == ""
    assert result.error is not None


async def test_non_string_content_degrades_gracefully_not_silently_stringified() -> None:
    """A valid-shaped response with `content: null` must not silently become
    the literal text "None" -- it should be treated as a failure."""

    def null_content(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": None}}]})

    client = _client(null_content)
    result = await client.extract_page(b"image", page_number=1)

    assert result.markdown == ""
    assert result.error is not None
    assert "None" not in result.markdown


async def test_client_error_status_is_not_retried() -> None:
    """400/401/403/404 are deterministic -- retrying them wastes attempts on
    something that can't succeed. Only 429/5xx are worth retrying."""
    calls = {"n": 0}

    def bad_request(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400)

    client = _client(bad_request, retry_max_attempts=5)
    result = await client.extract_page(b"image", page_number=1)

    assert calls["n"] == 1  # no retries attempted
    assert result.error is not None


async def test_sends_the_documented_request_shape() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = _client(handler)
    await client.extract_page(b"fake-image-bytes", page_number=1)

    body = captured["body"]
    assert isinstance(body, dict)
    message = body["messages"][0]
    assert message["role"] == "user"
    types = {part["type"] for part in message["content"]}
    assert types == {"image_url", "text"}


async def test_verify_succeeds_when_the_service_reports_ready() -> None:
    def healthy(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    client = _client(healthy)
    await client.verify()  # must not raise


async def test_verify_raises_when_the_service_is_unreachable() -> None:
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(unreachable)
    with pytest.raises(OCRServiceError, match="cannot reach"):
        await client.verify()


async def test_verify_raises_when_the_model_is_still_loading() -> None:
    def loading(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"status": "loading model"})

    client = _client(loading)
    with pytest.raises(OCRServiceError, match="not ready"):
        await client.verify()


async def test_verify_raises_on_an_unexpected_ready_body() -> None:
    def wrong_shape(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    client = _client(wrong_shape)
    with pytest.raises(OCRServiceError, match="not ready"):
        await client.verify()
